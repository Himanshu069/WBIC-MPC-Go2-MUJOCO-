"""
main_control_loop.py

Orchestrator that wires the existing pipeline (robot_model -> gait_scheduler
-> com_mpc -> finalqp -> prioritized_task_execution) into a live MuJoCo
simulation of the Go2.

"""

import os
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import time

from robot_model import PinModel
from gait_scheduler import GaitScheduler, Gaits, LEGS
from com_mpc import centroidal_mpc, N, dt as MPC_DT, DEFAULT_FOOT_POSITIONS
from finalqp import wbic_qp_solver_wbic
from prioritized_task_execution import PrioritizedTaskExecution
from foot_step_planner import FootStepPlanner 
from sim_plotter import SimulationLogger
import pinocchio as pin

XML_PATH = "unitree_go2/scene.xml"
WBIC_DT = 0.005         # 100 Hz (Fast whole-body control)
CTRL_DT = MPC_DT         # control loop step (50 Hz, matches com_mpc.dt)
SIM_END = 50.0          # seconds

n_fb = 6                # floating base DOFs (3 lin + 3 ang)
n_j = 12                # joint DOFs
n_q = n_fb + n_j        # 18 (velocity-space size)

VX_CMD      = 0.4
STEP_HEIGHT = 0.08 
def build_x_ref_traj(x0_vec: np.ndarray, x0_des: np.ndarray, vx_cmd: float) -> np.ndarray:
    x_ref = np.tile(x0_vec, (N, 1))
    for k in range(N):
        x_ref[k, 0]  = x0_des[0]   # Roll
        x_ref[k, 1]  = x0_des[1]   # Pitch
        x_ref[k, 2]  = x0_des[2]   # Yaw 
        # x_ref[k, 3]  = x0_des[3]
        x_ref[k, 4]  = x0_des[4]   # Y       
        x_ref[k, 5]  = x0_des[5]  # Z

        t_ahead = (k + 1) * CTRL_DT
        x_ref[k, 3]  = x0_vec[3] + vx_cmd * t_ahead
        x_ref[k, 6]   = 0.0   # ωx           
        x_ref[k, 7]   = 0.0   # ωy           
        x_ref[k, 9]   = vx_cmd
        x_ref[k, 10]  = 0.0
        x_ref[k, 11]  = 0.0
    return x_ref

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
    quat_mj = pin.Quaternion(w, x, y, z)

    quat_offset = pin.Quaternion(1.0, 0.0, 0.0, 0.0)

    quat_pin = quat_mj * quat_offset

    q = np.concatenate([
        qpos[0:3],     # base position, same convention
        quat_pin.coeffs(),      # reordered quaternion
        qpos[7:19],    # 12 joint angles, same order (FL,FR,RL,RR x hip,thigh,calf)
    ])

    dq = qvel.copy()  # (18,) base lin vel, base ang vel, 12 joint vels — no reorder needed

    return q, dq


# ONE CONTROL STEP
def compute_torques(robot, scheduler, q_curr, dq_curr, q0,
                    x0_des, planner,foot_pos_neutral, foot_pos_lift,
                    next_foot_positions, F_opt):
    """
    Run one MPC -> WBIC -> task-execution cycle and return joint torques
    (12,) in actuator order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh,
    FR_calf, RL_..., RR_... matching the MJCF <actuator> block exactly.
    """
    contact_now = scheduler.contact_state()            # (4,)

    g_vec, C_mat, M_full = robot.compute_dynamics_terms()

    tasks = []

    x_curr_f = np.zeros(12)
    x_des_f  = np.zeros(12)
    x_dot_des_f = np.zeros(12)
    x_ddot_des_f = np.zeros(12)         
    J_feet_full = np.zeros((12, 18))
    
    task_stance = {'J': [], 'x_curr': [], 'x_des': [], 'x_dot_des': [], 'x_ddot_des': [], 'Kp': [], 'Kd': []}
    task_swing  = {'J': [], 'x_curr': [], 'x_des': [], 'x_dot_des': [], 'x_ddot_des': [], 'Kp': [], 'Kd': []}
    
    for i, leg in enumerate(LEGS):
        foot_pos, _ = robot.get_single_foot_state_in_world(leg)
        J_foot = robot.compute_full_foot_Jacobian_world(leg)
        
        x_curr_f[3*i:3*i+3] = foot_pos
        J_feet_full[3*i:3*i+3, :] = J_foot
        
        if contact_now[i] == 1:   # STANCE
            task_stance['J'].append(J_foot)
            task_stance['x_curr'].append(foot_pos)
            task_stance['x_des'].append(foot_pos_neutral[leg])
            task_stance['x_dot_des'].append(np.zeros(3))
            task_stance['x_ddot_des'].append(np.zeros(3))
            task_stance['Kp'].append(np.array([1000.0, 1000.0, 1000.0]))
            task_stance['Kd'].append(np.array([100.0, 100.0, 100.0]))
        else:                      # SWING
            phase = scheduler.swing_phase(leg)
            p_land = next_foot_positions[leg]
            pos, vel, acc = planner.swing_foot_target(foot_pos_lift[leg], p_land, phase)
            
            x_des_f[3*i:3*i+3] = pos
            x_dot_des_f[3*i:3*i+3] = vel
            x_ddot_des_f[3*i:3*i+3] = acc
            
            task_swing['J'].append(J_foot)
            task_swing['x_curr'].append(foot_pos)
            task_swing['x_des'].append(pos)
            task_swing['x_dot_des'].append(vel)
            task_swing['x_ddot_des'].append(acc)
            task_swing['Kp'].append(np.array([1500.0, 1500.0, 1500.0])) # High gains stay!
            task_swing['Kd'].append(np.array([75.0, 75.0, 75.0]))

    # --- Base Task Definition ---

    rpy_curr = robot.current_config.rpy_world()
    R_curr = pin.rpy.rpyToMatrix(rpy_curr[0], rpy_curr[1], rpy_curr[2])

    roll, pitch, yaw = rpy_curr
    E = np.array([
    [1, np.sin(roll)*np.tan(pitch), np.cos(roll)*np.tan(pitch)],
    [0, np.cos(roll),              -np.sin(roll)              ],
    [0, np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)],
    ])
    # pos_err_world = x0_des[3:6] - q_curr[0:3]
    # pos_err_body = R_curr.T @ pos_err_world
    J_base = np.zeros((6, 18))
    J_base[0:3, 0:3] = np.eye(3)      # world lin vel -> body frame pos rate
    J_base[3:6, 3:6] = E   

    R_des = pin.rpy.rpyToMatrix(0.0, 0.0, 0.0) # Target flat orientation
    rot_err_body = pin.log3(R_curr.T @ R_des)
    v_des_world = np.array([VX_CMD, 0.0, 0.0])
    v_des_body = R_curr.T @ v_des_world
    q0[0] = q_curr[0] + VX_CMD * CTRL_DT 
    task_base = {
        'J': J_base, 
        'x_curr':  np.concatenate([q_curr[0:3], rpy_curr]),
        'x_des': np.concatenate([q0[0:3],     np.zeros(3)]),
        'x_dot_des': np.concatenate([v_des_body, np.zeros(3)]),
        'x_ddot_des': np.zeros(6),
        'Kp': np.array([1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0]), 
        'Kd': np.array([100.0,  100.0,  100.0,  200.0,  200.0,  200.0 ])
    }
    # print("x_curr", task_base['x_curr'])
    # print("x_des", task_base['x_des'])
    # print("x_dot_des", task_base['x_dot_des'])

    tasks = []
    
    tasks.append(task_base)
    if len(task_swing['J']) > 0:
        tasks.append({k: (np.vstack(v) if k == 'J' else np.concatenate(v)) for k, v in task_swing.items()})

    if len(task_stance['J']) > 0:
        tasks.append({k: (np.vstack(v) if k == 'J' else np.concatenate(v)) for k, v in task_stance.items()})        


    Jc_list = []
    for i, leg in enumerate(LEGS):
        if contact_now[i] == 1:
            Jc_list.append(robot.compute_full_foot_Jacobian_world(leg))
    Jc = np.vstack(Jc_list) if Jc_list else np.zeros((0, 18))
    
    pte = PrioritizedTaskExecution(n_dof=18)
    q_cmd, q_dot_cmd, q_ddot_cmd = pte.execute(tasks, q_curr, dq_curr, A=M_full)
    
    if Jc is None:
        Jc = np.zeros((0, 18))

    b_vec = C_mat @ dq_curr

    Sf = np.hstack([np.eye(n_fb), np.zeros((n_fb, n_j))])  # (6, 18)

 

    fr_MPC_list = []
    for i in range(4):
        if contact_now[i] == 1:
            fr_MPC_list.append(F_opt[3*i:3*i+3, 0])
    fr_MPC = np.concatenate(fr_MPC_list) if fr_MPC_list else np.array([]) 
    # print(f"Jc shape: {Jc.shape}, fr_MPC shape: {fr_MPC.shape}")

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
        A=M_full, b=b_vec, g=g_vec,
        Jc=Jc, Sf=Sf,
        fr_MPC=fr_MPC,
        q_ddot_cmd=q_ddot_cmd,
        W=W, n_j=n_j,
    )

    Sa = np.hstack([np.zeros((n_j, n_fb)), np.eye(n_j)])  # (12, 18)
    q_ddot_wbic = q_ddot_cmd + np.concatenate([delta_f, np.zeros(12)])    # (18,)
    # print("q_ddot_wbic",q_ddot_wbic)
    full_tau = (M_full @ q_ddot_wbic + b_vec + g_vec - Jc.T @ fr_opt)
    tau = Sa @ full_tau

    f_wbic_all = np.zeros(12)
    active_contact_idx = 0
    for i in range(4):
        if contact_now[i] == 1:
            f_wbic_all[3*i : 3*i+3] = fr_opt[3*active_contact_idx : 3*active_contact_idx+3]
            active_contact_idx += 1

    q_j_curr = q_curr[7:19]      # current joint positions
    q_dot_j_curr = dq_curr[6:18] # current joint velocities

    q_cmd_j = q_cmd              # 12-dim from PTE
    q_dot_cmd_j = q_dot_cmd[6:18] # joint portion of velocity command

    Kp_pd = 10 # tune these
    Kd_pd = 1.0

    tau_pd = Kp_pd * (q_cmd_j - q_j_curr) + Kd_pd * (q_dot_cmd_j - q_dot_j_curr)
    tau = tau + tau_pd
    return tau, f_wbic_all, x_curr_f, x_des_f


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
    scheduler = GaitScheduler(Gaits.trot(), dt=WBIC_DT)
    foot_planner = FootStepPlanner(robot, v_cmd=np.array([VX_CMD, 0.0, 0.0]))

    q0, dq0 = mujoco_to_pin_state(data)
    robot.update_model(q0, dq0)

    for leg in LEGS:
        pos, _ = robot.get_single_foot_state_in_world(leg)
        print(f"{leg}: {pos}")
    print(f"base z: {q0[2]}")

    curr_contact = scheduler.contact_state()

    x0_des = robot.compute_com_x_vec().flatten()
    foot_pos_neutral = {leg: robot.get_single_foot_state_in_world(leg)[0].copy() for leg in LEGS}
    foot_pos_lift    = {leg: foot_pos_neutral[leg].copy() for leg in LEGS}
    prev_contact     = scheduler.contact_state().copy()

    joint_body_ids = []
    leg_prefixes = ['FL', 'FR', 'RL', 'RR']
    joint_suffixes = ['hip', 'thigh', 'calf']
    for leg in leg_prefixes:
        for joint in joint_suffixes:
            b_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{leg}_{joint}")
            if b_id == -1: 
                b_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{leg.lower()}_{joint}")
            joint_body_ids.append(b_id)

    logger = SimulationLogger()
    # ---- visualization setup ----
    glfw.init()
    window = glfw.create_window(1200, 900, "Go2 WBIC-MPC", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    cam.azimuth, cam.elevation, cam.distance = -90, -20, 1.5
    cam.lookat = np.array([0.0, 0.0, 0.3])

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    last_mpc_time = -MPC_DT
    last_wbic_time = -WBIC_DT
    last_render_time = 0.0

    # Store the latest MPC forces globally so the WBIC can use them between MPC updates
    latest_F_opt = np.zeros((12, 1)) 
    tau = np.zeros(n_j)
    pushed = False
    while not glfw.window_should_close(window) and data.time < SIM_END:
        # Run the (slower) control loop only every CTRL_DT seconds —
        # MPC/WBIC are too expensive to solve every 1ms physics step.

        if data.time - last_mpc_time >= CTRL_DT:  # CTRL_DT is 0.02s (50 Hz)
            last_mpc_time = data.time

            # Update models to find foot locations for optimization
            q_mpc, dq_mpc = mujoco_to_pin_state(data)
            robot.update_model(q_mpc, dq_mpc)

            x0_vec = robot.compute_com_x_vec().flatten()
            x_ref_traj = build_x_ref_traj(x0_vec, x0_des, VX_CMD)
                    
            ratio = int(CTRL_DT / WBIC_DT)
            contact_schedule = scheduler.contact_schedule(N * ratio)[::ratio]

            FL_r, FR_r, RL_r, RR_r = robot.get_foot_lever_world()
            live_foot_positions = {"FL": FL_r, "FR": FR_r, "RL": RL_r, "RR": RR_r}
            
            try:
                _, latest_F_opt = centroidal_mpc(
                    x0_vec, x_ref_traj, contact_schedule,
                    foot_positions=live_foot_positions,
                )
            except Exception as e:
                print(f"[MPC] optimization failed at t={data.time:.3f}: {e}")

        if data.time - last_wbic_time >= WBIC_DT:  # WBIC_DT is 0.002s (500 Hz)
            last_wbic_time = data.time

            # Update robot kinematic matrices with the absolute latest state
            q, dq = mujoco_to_pin_state(data)
            robot.update_model(q, dq)
            
            scheduler.step()
            next_foot_positions = foot_planner.compute_next_foot_positions()
            curr_contact = scheduler.contact_state()
            
            for i, leg in enumerate(LEGS):
                if prev_contact[i] == 1 and curr_contact[i] == 0:
                    foot_pos_lift[leg], _ = robot.get_single_foot_state_in_world(leg)
                    foot_pos_lift[leg] = foot_pos_lift[leg].copy()
                if prev_contact[i] == 0 and curr_contact[i] == 1:
                    actual_pos, _ = robot.get_single_foot_state_in_world(leg)
                    foot_pos_neutral[leg] = actual_pos.copy()
                    foot_pos_neutral[leg][2] = 0.0
                    
            prev_contact = curr_contact.copy()

            try:
                # Track the base and feet references using cached latest_F_opt forces
                # t0 = time.perf_counter()
                tau, f_eff, x_curr_f, x_des_f = compute_torques(
                    robot, scheduler, q, dq, q0,
                    x0_des, foot_planner, foot_pos_neutral, foot_pos_lift,
                    next_foot_positions, latest_F_opt
                )
                # print("x_curr_f", x_curr_f)
                # print("x_des_f", x_des_f)

                # elapsed = (time.perf_counter() - t0) * 1000
                # if elapsed > 2.0:
                #     print(f"[WBIC slow] {elapsed:.2f} ms at t={data.time:.3f}")

            except Exception as e:
                print(f"[WBIC] torque calculation failed at t={data.time:.3f}: {e}") 
                tau = np.zeros(12)
                f_eff = np.zeros(12)
                x_curr_f, x_des_f = np.zeros(12), np.zeros(12)

            # --- High Resolution Simulation Data Logging ---
            j_bearing_forces = np.zeros(12)
            for idx, b_id in enumerate(joint_body_ids):
                if b_id != -1:
                    f_xyz = data.cfrc_int[b_id, 3:6]
                    j_bearing_forces[idx] = np.linalg.norm(f_xyz)

            actual_joints = q[7:19]
            desired_joints = q0[7:19] 
            joint_errors = np.abs(desired_joints - actual_joints)

            logger.log(
                data.time, 
                q[0:3], q0[0:3], 
                robot.current_config.rpy_world(), np.zeros(3), 
                dq[0:3].copy(), np.array([VX_CMD, 0.0, 0.0]),
                tau, f_eff, j_bearing_forces,
                x_curr_f[[2, 5, 8, 11]], x_des_f[[2, 5, 8, 11]], joint_errors
            )
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        if data.time - last_render_time >= (1.0 / 60.0):
            last_render_time = data.time
            
            cam.lookat[:] = data.qpos[0:3]
            # cam.azimuth, cam.elevation, cam.distance = -135, -30, 2.0            
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

    print("\n[Simulation] Window closed or time limit hit. Processing plot profiles...")
    logger.plot()


if __name__ == "__main__":
    main()