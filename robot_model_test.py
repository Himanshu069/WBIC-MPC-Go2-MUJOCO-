import numpy as np
from com_mpc import centroidal_mpc, foot_positions, foot_names, N
from robot_model import PinModel, ConfigurationState
from foot_step_planner import FootStepPlanner
from prioritized_task_execution import PrioritizedTaskExecution
from finalqp import wbic_qp_solver_wbic

# -----------------------------
# 1️⃣ Initialize robot model
# -----------------------------
robot = PinModel()
q_curr = robot.q_init.copy()
dq_curr = robot.dq_init.copy()

# -----------------------------
# 2️⃣ Footstep planner
# -----------------------------
planner = FootStepPlanner(robot)
next_foot_positions = planner.compute_next_foot_positions()
print("Next foot positions:", next_foot_positions)

# -----------------------------
# 3️⃣ Centroidal MPC
# -----------------------------
# Dummy reference trajectory (flat, constant height)
x0_vec = robot.compute_com_x_vec()
if x0_vec.ndim == 2 and x0_vec.shape[1] == 1:
    x0_vec = x0_vec.flatten()

x_ref_traj = np.tile(x0_vec, (N,1))
# Dummy contact schedule: all feet stance
contact_schedule = np.ones((N, 4))  

X_opt, F_opt = centroidal_mpc(robot.compute_com_x_vec().flatten(), x_ref_traj, contact_schedule)
print("MPC COM trajectory shape:", X_opt.shape)
print("MPC reaction forces shape:", F_opt.shape)

# -----------------------------
# 4️⃣ Prioritized task execution
# -----------------------------
pte = PrioritizedTaskExecution(n_joints=12)

# Example tasks: maintain body orientation and keep feet at MPC positions
tasks = []

# 4a) Body orientation (3x12 Jacobian selecting angular part)
J_body = np.zeros((3, 12))
J_body[:,3:6] = np.eye(3)
x_des_body = X_opt[3:6,-1].flatten()  # last reference orientation
tasks.append({'J': J_body, 'x_des': x_des_body})

# 4b) Foot positions (each 3x12 Jacobian in world frame)
J_full = np.zeros((12, 12))       # 12x12: 3 DoF per foot × 4 feet
x_des_full = np.zeros(12)         # desired foot positions for all legs

foot_names = ['FL', 'FR', 'RL', 'RR']

for i, leg in enumerate(foot_names):
    # Get full 3xnv Jacobian for this foot in world frame
    J_foot_world = robot.compute_full_foot_Jacobian_world(leg)  # 3 x nv

    # Extract only the columns corresponding to the leg joints
    joint_ids = [
        robot.model.getJointId(f"{leg}_hip_joint"),
        robot.model.getJointId(f"{leg}_thigh_joint"),
        robot.model.getJointId(f"{leg}_calf_joint"),
    ]
    vcols = [robot.model.joints[jid].idx_v for jid in joint_ids]

    J_leg = J_foot_world[:, vcols]   # 3x3 for this leg

    # Place into block-diagonal of full 12x12 Jacobian
    J_full[3*i:3*i+3, 3*i:3*i+3] = J_leg

    # Desired foot positions (proxy using MPC foot forces, or real positions)
    x_des_full[3*i:3*i+3] = F_opt[3*i:3*i+3, -1]

# Now tasks has only one element with full Jacobian and full foot positions
tasks = [{'J': J_full, 'x_des': x_des_full}]
# Execute prioritized tasks
A_dummy = np.eye(12)  # inertia matrix placeholder
# print("J",tasks[0]["J"] )
# print("J",tasks[1]["J"] )

q_cmd, q_dot_cmd, q_ddot_cmd = pte.execute(tasks, q_curr[7:], dq_curr[6:], A=A_dummy)
print("Desired joint positions:", q_cmd)
print("Desired joint accelerations:", q_ddot_cmd)

# -----------------------------
# 5️⃣ WBIC QP (final reaction forces)
# -----------------------------
# Build dummy matrices for WBIC QP
n_fb = 6
n_j = 12
n_q = n_fb + n_j  # 18 total

A_q = np.eye(n_q)         # 18x18
b_q = np.zeros(n_q)       # 18
g_q = np.zeros(n_q)       # 18
Jc_dummy = np.zeros((12, n_q))  # 12 contact forces, 18x12 Jacobian transpose
Sf_dummy = np.hstack([np.eye(n_fb), np.zeros((n_fb, n_j))])  # 6x18
W_dummy = np.eye(12)  # inequality on 12 contact forces

fr_MPC = F_opt[:, -1]
fr_opt, delta_fr_opt, delta_f_opt = wbic_qp_solver_wbic(
    A=A_q, b=b_q, g=g_q, Jc=Jc_dummy, Sf=Sf_dummy,
    fr_MPC=fr_MPC.flatten(), q_ddot_cmd=q_ddot_cmd.flatten(),
    W=W_dummy, n_j=n_j
)
print("Optimized reaction forces:", fr_opt)
print("Delta fr:", delta_fr_opt)
print("Delta floating base acceleration:", delta_f_opt)

# -----------------------------
# 6️⃣ Update robot model
# -----------------------------
robot.update_model(q_cmd, q_dot_cmd)
FL, FR, RL, RR = robot.get_foot_placement_in_world()
print("Foot placements in world:", FL, FR, RL, RR)
