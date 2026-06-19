import numpy as np
from robot_model import PinModel

robot = PinModel()

print("=== Model sanity check ===")
print(f"nq (config space): {robot.model.nq}")   # should be 19
print(f"nv (vel space):    {robot.model.nv}")   # should be 18

q = robot.q_init
dq = robot.dq_init
print(f"\nq_init shape:  {q.shape}")            # (19,)
print(f"dq_init shape: {dq.shape}")             # (18,)

FL, FR, RL, RR = robot.get_foot_placement_in_world()
print(f"\nFoot positions at init:")
print(f"  FL: {np.round(FL, 4)}")
print(f"  FR: {np.round(FR, 4)}")
print(f"  RL: {np.round(RL, 4)}")
print(f"  RR: {np.round(RR, 4)}")
# All z should be near 0 (ground), x/y should be symmetric

com = robot.pos_com_world
print(f"\nCoM position: {np.round(com, 4)}")    # z should be ~0.27

x_vec = robot.compute_com_x_vec().flatten()
print(f"\n12-DOF state vector: \n {np.round(x_vec, 4)}")
# First 3: CoM pos, next 3: RPY (~0), next 3: vel (~0), last 3: omega (~0)

# Jacobian shape check
from gait_scheduler import LEGS
for leg in LEGS:
    J = robot.compute_full_foot_Jacobian_world(leg)
    print(f"  J_{leg} shape: {J.shape}")        # should be (3, 18)