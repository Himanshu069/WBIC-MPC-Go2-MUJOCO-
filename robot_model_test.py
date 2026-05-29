import numpy as np
from gait_scheduler import GaitScheduler, Gaits, LEGS
from com_mpc import centroidal_mpc, N
from robot_model import PinModel
from foot_step_planner import FootStepPlanner
from prioritized_task_execution import PrioritizedTaskExecution
from finalqp import wbic_qp_solver_wbic
import pinocchio as pin

# ── 1. Initialize ─────────────────────────────────────────────────────────────
robot     = PinModel()
q_curr    = robot.q_init.copy()    # (19,)
dq_curr   = robot.dq_init.copy()   # (18,)
planner   = FootStepPlanner(robot)
pte       = PrioritizedTaskExecution(n_joints=12)
scheduler = GaitScheduler(Gaits.trot(period=0.5), dt=0.02)

n_fb = 6
n_j  = 12
n_q  = n_fb + n_j  # 18

# ── 2. Gait state ─────────────────────────────────────────────────────────────
scheduler.step()
contact_now      = scheduler.contact_state()       # (4,)
contact_schedule = scheduler.contact_schedule(N)   # (N, 4)

# ── 3. Reference trajectory (hold current state) ──────────────────────────────
x0_vec     = robot.compute_com_x_vec().flatten()   # (12,)
x_ref_traj = np.tile(x0_vec, (N, 1))              # (N, 12)

# ── 4. Live foot lever arms for MPC B matrix ──────────────────────────────────
# These are r vectors (foot pos relative to CoM) in world frame
FL_r, FR_r, RL_r, RR_r = robot.get_foot_lever_world()
live_foot_positions = {
    "FL": FL_r, "FR": FR_r, "RL": RL_r, "RR": RR_r
}

X_opt, F_opt = centroidal_mpc(
    x0_vec,
    x_ref_traj,
    contact_schedule,
    foot_positions=live_foot_positions,   # ← pass live positions in
)
print("MPC COM trajectory:", X_opt.shape)
print("MPC forces:        ", F_opt.shape)

# ── 5. Prioritized task execution ─────────────────────────────────────────────
# Get real mass matrix from Pinocchio (nv x nv = 18 x 18)
_, _, M_full = robot.compute_dynamics_terms()

# Build tasks with correct 3×18 Jacobians
tasks = []

# Task 1 (highest priority): body orientation
# Angular Jacobian of the base — rows 3:6 of the 6D base Jacobian
J_orient = np.zeros((3, n_q))
J_orient[:, 3:6] = np.eye(3)   # base angular velocity columns
x_des_orient = x0_vec[3:6]     # hold current roll/pitch/yaw
tasks.append({
    'J':       J_orient,
    'x_des':   x_des_orient,
    'Kp':      50.0,
    'Kd':      5.0,
})

# Task 2: foot positions (stance feet only — swing feet handled by trajectory generator)
J_feet  = np.zeros((12, n_q))
x_des_f = np.zeros(12)

for i, leg in enumerate(LEGS):
    # Full 3×nv Jacobian from Pinocchio
    J_foot_full = robot.compute_full_foot_Jacobian_world(leg)  # (3, 18)
    J_feet[3*i:3*i+3, :] = J_foot_full

    # Desired position = current foot position (hold stance feet in place)
    foot_pos, _ = robot.get_single_foot_state_in_world(leg)
    x_des_f[3*i:3*i+3] = foot_pos

tasks.append({
    'J':     J_feet,
    'x_des': x_des_f,
    'Kp':    20.0,
    'Kd':    2.0,
})

# PTE operates on joint space only (12,) — base handled separately by WBIC
q_joints  = q_curr[7:]    # (12,)  joint angles
dq_joints = dq_curr[6:]   # (12,)  joint velocities
A_joints  = M_full[6:, 6:]  # (12, 12) joint-space inertia block

q_cmd, q_dot_cmd, q_ddot_cmd = pte.execute(
    tasks, q_joints, dq_joints, A=A_joints
)

# ── 6. Build real WBIC matrices from Pinocchio ────────────────────────────────
g_vec, C_mat, M_mat = robot.compute_dynamics_terms()

# b = C * dq (Coriolis/centrifugal torques)
b_vec = C_mat @ dq_curr   # (18,)

# Floating-base selector: picks rows 0:6 (unactuated DOFs)
Sf = np.hstack([np.eye(n_fb), np.zeros((n_fb, n_j))])  # (6, 18)

# Contact Jacobian: stack foot Jacobians for stance feet only
Jc_list = []
for i, leg in enumerate(LEGS):
    if contact_now[i] == 1:
        J_foot = robot.compute_full_foot_Jacobian_world(leg)  # (3, 18)
        Jc_list.append(J_foot)

if len(Jc_list) > 0:
    Jc = np.vstack(Jc_list)   # (3*n_stance, 18)
else:
    Jc = np.zeros((1, n_q))

# Full q_ddot command (floating base gets zeros — WBIC will solve for it)
q_ddot_full = np.concatenate([np.zeros(n_fb), q_ddot_cmd])  # (18,)

# fr_MPC: only stance feet have forces
fr_MPC_list = []
for i in range(4):
    if contact_now[i] == 1:
        fr_MPC_list.append(F_opt[3*i:3*i+3, 0])

fr_MPC = np.concatenate(fr_MPC_list) if fr_MPC_list else np.zeros(3)

# Friction cone inequality matrix (identity for now — you'll expand this)
n_stance = len(Jc_list)
W = np.eye(3 * n_stance)

fr_opt, delta_fr, delta_f = wbic_qp_solver_wbic(
    A=M_mat, b=b_vec, g=g_vec,
    Jc=Jc, Sf=Sf,
    fr_MPC=fr_MPC,
    q_ddot_cmd=q_ddot_full,
    W=W, n_j=n_j
)
print("Optimized contact forces:", fr_opt)

# ── 7. Compute joint torques ──────────────────────────────────────────────────
# τ = M q̈ + C q̇ + g - Jcᵀ fc   (projected to actuated joints)
# Actuator selector: rows 6:18
Sa = np.hstack([np.zeros((n_j, n_fb)), np.eye(n_j)])  # (12, 18)

q_ddot_wbic = np.concatenate([delta_f, q_ddot_cmd])  # (18,)
tau = Sa @ (M_mat @ q_ddot_wbic + b_vec + g_vec - Jc.T @ fr_opt)
print("Joint torques:", tau)

# ── 8. Update model with full-size q, dq ─────────────────────────────────────
# Reconstruct full q (19,) from base state + new joint commands
q_new = q_curr.copy()
q_new[7:] = q_cmd          # update joint angles only

dq_new = dq_curr.copy()
dq_new[6:] = q_dot_cmd     # update joint velocities only

robot.update_model(q_new, dq_new)
FL, FR, RL, RR = robot.get_foot_placement_in_world()
print("Foot placements:", FL, FR, RL, RR)