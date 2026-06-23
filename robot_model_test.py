import numpy as np
from robot_model import PinModel
from gait_scheduler import GaitScheduler , Gaits, LEGS
from com_mpc import centroidal_mpc, N, DEFAULT_FOOT_POSITIONS


robot = PinModel()
print("Robot Total Mass:", robot.total_mass)
x0_vec = robot.compute_com_x_vec().flatten()        # (12,)

x_ref_traj = np.tile(x0_vec, (N, 1))                # (N, 12)
scheduler = GaitScheduler(
    Gaits.stand(),
    dt=0.03
)
scheduler.step()
contact_now = scheduler.contact_state()            # (4,)
contact_schedule = scheduler.contact_schedule(N) 
FL_r, FR_r, RL_r, RR_r = robot.get_foot_lever_world()
live_foot_positions = {"FL": FL_r, "FR": FR_r, "RL": RL_r, "RR": RR_r}
X_opt, F_opt = centroidal_mpc(
    x0_vec, x_ref_traj, contact_schedule,
    foot_positions=live_foot_positions,
)

# print("=== State trajectory (per horizon step) ===")
# for k in range(X_opt.shape[1]):          # N+1 steps
#     print(f"x_{k} =", X_opt[:, k])

# print("\n=== Force trajectory (per horizon step) ===")
# for k in range(F_opt.shape[1]):          # N steps
#     print(f"f_{k} =", F_opt[:, k])