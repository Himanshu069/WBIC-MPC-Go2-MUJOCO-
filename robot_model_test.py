import numpy as np
import pinocchio as pin
from robot_model import PinModel
from gait_scheduler import GaitScheduler , Gaits, LEGS
from com_mpc import centroidal_mpc, N, DEFAULT_FOOT_POSITIONS
from prioritized_task_execution import PrioritizedTaskExecution

robot = PinModel()
print("Robot Total Mass:", robot.total_mass)

scheduler = GaitScheduler(
    Gaits.stand(),
    dt=0.03
)
scheduler.step()
contact_schedule = scheduler.contact_schedule(N) 

q_curr = robot.current_config.q()     
dq_curr = robot.current_config.dq()

x0_vec = robot.compute_com_x_vec().flatten()        # (12,)
foot_pos_des = {leg: robot.get_single_foot_state_in_world(leg)[0].copy() for leg in LEGS}
base_pos_des = q_curr[0:3].copy()
x_ref_traj = np.tile(x0_vec, (N, 1))                # (N, 12)

FL_r, FR_r, RL_r, RR_r = robot.get_foot_lever_world()
live_foot_relative = {"FL": FL_r, "FR": FR_r, "RL": RL_r, "RR": RR_r}
X_opt, F_opt = centroidal_mpc(
    x0_vec, x_ref_traj, contact_schedule,
    foot_positions=live_foot_relative,
)

print("=== State trajectory (per horizon step) ===")
for k in range(X_opt.shape[1]):          # N+1 steps
    print(f"x_{k} =", X_opt[:, k])

print("\n=== Force trajectory (per horizon step) ===")
for k in range(F_opt.shape[1]):          # N steps
    print(f"f_{k} =", F_opt[:, k])

FL_p, FR_p, RL_p, RR_p = robot.get_foot_placement_in_world()
live_foot_positions = {"FL": FL_p, "FR": FR_p, "RL": RL_p, "RR": RR_p}

_, _, M_full = robot.compute_dynamics_terms()

tasks = []
J_pos = np.hstack([np.eye(3), np.zeros((3, 15))])   
tasks.append({
        'name': 'Base_Position',
        'J': J_pos,  
        'x_curr': q_curr[0:3], 
        'x_des': base_pos_des, 
        'x_dot_des': np.zeros(3),   # Added safe zeros for derivative terms
        'x_ddot_des': np.zeros(3),
        'Kp': 50.0, 
        'Kd': 10.0
    })

x_des_f = np.zeros(12)
x_curr_f = np.zeros(12)
J_feet_full = np.zeros((12, 18))

for i, leg in enumerate(LEGS):
        foot_pos, _ = robot.get_single_foot_state_in_world(leg)
        x_curr_f[3*i:3*i+3] = foot_pos
        x_des_f[3*i:3*i+3] = foot_pos_des[leg]
        J_feet_full[3*i:3*i+3, :] = robot.compute_full_foot_Jacobian_world(leg)

tasks.append({
        'name': 'All_Feet_Position',
        'J': J_feet_full,
        'x_curr': x_curr_f, 
        'x_des': x_des_f, 
        'x_dot_des': np.zeros(12),  # Added safe zeros for derivative terms
        'x_ddot_des': np.zeros(12),
        'Kp': 20.0, 
        'Kd': 2.0
    })

    # --- 5. Execute Prioritized Tasks ---
pte = PrioritizedTaskExecution(n_dof=18)
q_cmd, q_dot_cmd, q_ddot_cmd = pte.execute(tasks, q_curr, dq_curr, A=M_full)

# --- 6. Verify Outputs ---
print("\n=== Prioritized Task Execution (PTE) Outputs ===")

print("q_cmd (Desired Positions):")
print(np.round(q_cmd, 4))

print("\nq_dot_cmd (Desired Velocities):")
print(np.round(q_dot_cmd, 4))

print("\nq_ddot_cmd (Desired Accelerations):")
print(np.round(q_ddot_cmd, 4))

