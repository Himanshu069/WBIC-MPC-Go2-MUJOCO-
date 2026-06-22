"""
main_control_loop.py

Orchestrator that wires the existing pipeline (robot_model -> gait_scheduler
-> com_mpc -> finalqp -> prioritized_task_execution) into a live MuJoCo
simulation of the Go2.

MILESTONE: standing balance only. No swing-leg trajectories, no velocity
commands yet. x_ref_traj holds the robot at its current pose. Once this is
stable, swing-foot tasks + a real reference generator get added on top.

"""

import os
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import time

from robot_model import PinModel
from gait_scheduler import GaitScheduler, Gaits, LEGS
from com_mpc import centroidal_mpc, N, DEFAULT_FOOT_POSITIONS
from finalqp import wbic_qp_solver_wbic
from prioritized_task_execution import PrioritizedTaskExecution

XML_PATH = "unitree_go2/scene.xml"
SIM_DT = 0.001          # MuJoCo physics step
CTRL_DT = 0.03          # control loop step (50 Hz, matches com_mpc.dt)
SIM_END = 50.0          # seconds

n_fb = 6                # floating base DOFs (3 lin + 3 ang)
n_j = 12                # joint DOFs
n_q = n_fb + n_j        # 18 (velocity-space size)


#  STATE BRIDGE: MuJoCo (data.qpos/qvel) -> Pinocchio (q, dq)

def mujoco_to_pin_state(data: "mj.MjData") -> tuple[np.ndarray, np.ndarray]:
    """
    Convert live MuJoCo state into the (q, dq) convention used by
    robot_model.PinModel / ConfigurationState.

    The ONLY structural difference between MuJoCo's qpos/qvel and Pinocchio's
    q/dq for this free-flyer + 12 revolute-joint robot is quaternion order:

        MuJoCo  qpos[3:7] = [w, x, y, z]
        Pinocchio q[3:7]  = [x, y, z, w]

    Positions (xyz), joint angles, linear velocity, and angular velocity are
    already in matching order/frame convention (MuJoCo's qvel base angular
    velocity is body-frame, same as Pinocchio's free-flyer convention).
    """
    qpos = data.qpos  # (19,) for free-flyer + 12 joints
    qvel = data.qvel  # (18,)

    # --- quaternion reorder: wxyz -> xyzw ---
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    pin_quat = np.array([x, y, z, w])

    q = np.concatenate([
        qpos[0:3],     # base position, same convention
        pin_quat,      # reordered quaternion
        qpos[7:19],    # 12 joint angles, same order (FL,FR,RL,RR x hip,thigh,calf)
    ])

    dq = qvel.copy()  # (18,) base lin vel, base ang vel, 12 joint vels — no reorder needed

    return q, dq


# ONE CONTROL STEP
def compute_torques(robot: PinModel, scheduler: GaitScheduler,
                     q_curr: np.ndarray, dq_curr: np.ndarray) -> np.ndarray:
    """
    Run one MPC -> WBIC -> task-execution cycle and return joint torques
    (12,) in actuator order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh,
    FR_calf, RL_..., RR_... matching the MJCF <actuator> block exactly.
    """
    scheduler.step()
    contact_now = scheduler.contact_state()            # (4,)
    contact_schedule = scheduler.contact_schedule(N)    # (N, 4)

    x0_vec = robot.compute_com_x_vec().flatten()        # (12,)
    x_ref_traj = np.tile(x0_vec, (N, 1))                # (N, 12)

    FL_r, FR_r, RL_r, RR_r = robot.get_foot_lever_world()
    live_foot_positions = {"FL": FL_r, "FR": FR_r, "RL": RL_r, "RR": RR_r}
    X_opt, F_opt = centroidal_mpc(
        x0_vec, x_ref_traj, contact_schedule,
        foot_positions=live_foot_positions,
    )

    g_vec, C_mat, M_full = robot.compute_dynamics_terms()

    tasks = []
    J_pos = np.hstack([np.eye(3), np.zeros((3, 15))])   # (3,18)
    x_des_pos = q_curr[0:3]   # current position
    tasks.append({'J': J_pos,  'x_curr': q_curr[0:3], 'x_des': x_des_pos, 'Kp': 50.0, 'Kd': 10.0})
    

    x_des_f = np.zeros(12)
    x_curr_f = np.zeros(12)
    J_feet_full = np.zeros((12, 18))
    for i, leg in enumerate(LEGS):
        foot_pos, _ = robot.get_single_foot_state_in_world(leg)
        x_curr_f[3*i:3*i+3] = foot_pos
        x_des_f[3*i:3*i+3] = foot_pos
        J_full = robot.compute_full_foot_Jacobian_world(leg)   # (3,18)
        J_feet_full[3*i:3*i+3, :] = J_full

    tasks.append({'J': J_feet_full,'x_curr': x_curr_f, 'x_des': x_des_f, 'Kp': 20.0, 'Kd': 2.0})

    pte = PrioritizedTaskExecution(n_dof=18)
    q_cmd, q_dot_cmd, q_ddot_cmd = pte.execute(tasks, q_curr, dq_curr, A=M_full)

    b_vec = C_mat @ dq_curr

    Sf = np.hstack([np.eye(n_fb), np.zeros((n_fb, n_j))])  # (6, 18)

    Jc_list = []
    for i, leg in enumerate(LEGS):
        if contact_now[i] == 1:
            Jc_list.append(robot.compute_full_foot_Jacobian_world(leg))
    Jc = np.vstack(Jc_list) if Jc_list else np.zeros((1, n_q))


    fr_MPC_list = []
    for i in range(4):
        if contact_now[i] == 1:
            fr_MPC_list.append(F_opt[3*i:3*i+3, 0])
    fr_MPC = np.concatenate(fr_MPC_list) if fr_MPC_list else np.zeros(3)


    mu = 0.6
    nc = len(Jc_list)  
    if nc > 0:
        W = np.zeros((5 * nc, 3 * nc))
        for i in range(nc):
            W[5*i:5*i+5, 3*i:3*i+3] = np.array([
                [1.0, 0.0,  mu],
                [-1.0, 0.0, mu],
                [0.0, 1.0,  mu],
                [0.0, -1.0, mu],
                [0.0, 0.0,  1.0]
            ])
    else:
        W = np.eye(1) 
    
    fr_opt, delta_fr, delta_f = wbic_qp_solver_wbic(
        A=M_mat, b=b_vec, g=g_vec,
        Jc=Jc, Sf=Sf,
        fr_MPC=fr_MPC,
        q_ddot_cmd=q_ddot_full,
        W=W, n_j=n_j,
    )

    Sa = np.hstack([np.zeros((n_j, n_fb)), np.eye(n_j)])  # (12, 18)
    q_ddot_wbic = q_ddot_cmd + np.concatenate([delta_f, np.zeros(12)])    # (18,)
    print("q_ddot_wbic",q_ddot_wbic)
    full_tau = (M_mat @ q_ddot_wbic + b_vec + g_vec - Jc.T @ fr_opt)
    tau = Sa @ full_tau
    return tau


#  MAIN LOOP
def main():
    dirname = os.path.dirname(__file__)
    xml_path = os.path.join(dirname, XML_PATH)

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # Start from the MJCF's "home" keyframe (standing pose) instead of zeros.
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mj.mj_resetDataKeyframe(model, data, key_id)
    else:
        mj.mj_resetData(model, data)

    mj.mj_forward(model, data)

    robot = PinModel()
    scheduler = GaitScheduler(Gaits.stand(), dt=CTRL_DT)

    # ---- visualization setup ----
    glfw.init()
    window = glfw.create_window(1200, 900, "Go2 WBIC-MPC", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    cam.azimuth, cam.elevation, cam.distance = -90, -20, 1.5
    cam.lookat = np.array([0.0, 0.0, 0.3])

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    last_ctrl_time = -CTRL_DT  # forces a control update on the first step
    tau = np.zeros(n_j)

    while not glfw.window_should_close(window) and data.time < SIM_END:
        # Run the (slower) control loop only every CTRL_DT seconds —
        # MPC/WBIC are too expensive to solve every 1ms physics step.
        if data.time - last_ctrl_time >= CTRL_DT:
            last_ctrl_time = data.time

            q, dq = mujoco_to_pin_state(data)
            robot.update_model(q, dq)
            # print(f"rpy_world: {robot.current_config.rpy_world()}")
            try:
                tau = compute_torques(robot, scheduler, q, dq)
            except Exception as e:
                print(f"[control] solver failed at t={data.time:.3f}: {e}")
                # Fail-safe: hold last known good torque rather than crash sim.
        # print("tau", tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        # ---- render ----
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()