# WBIC-MPC Go2 — MuJoCo Simulation

A full hierarchical locomotion controller for the **Unitree Go2** quadruped robot, implemented in Python and validated in MuJoCo. The stack combines a **centroidal Model Predictive Controller (MPC)** for high level planning with a **Whole Body Impulse Controller (WBIC)** for low-level joint command generation via prioritized task execution. This is a direct implementatin of the paper: https://arxiv.org/pdf/1909.06586

---

Trot Gait Example:


https://github.com/user-attachments/assets/109676e8-ef3e-42d5-8d04-4db0877d0357

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8239ffd5-83e0-4134-8718-5e9dcfbe9db2" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/627f894b-7005-4a1e-bba5-50372c48f35a" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/6b6baa58-3352-40f8-a594-b2da95997077" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0eacf4b0-1134-4bd4-a73d-d9867925a546"" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/13dda594-6d00-4f3b-bf59-3fbec95730ec" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/0599d35d-d24f-4741-ac86-65f4f2ed9d19" width="100%"/></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><img src="https://github.com/user-attachments/assets/8d1c5d74-e532-4007-a88e-7254d08b5088" width="33%"/></td>
  </tr>
</table>


## Modules

| File | Description |
|---|---|
| `gait_scheduler.py` | Phase-based gait state machine. Supports trot, walk, pace, bound, pronk, stand, and arbitrary custom gaits. Provides `contact_state()` for WBIC and `contact_schedule(N)` for the MPC horizon. |
| `com_mpc.py` | Centroidal MPC over a 12-D state `[θ, p, ω, v]` using a linearised single-rigid-body model. Solved a QP in Casadi with friction pyramid and unilateral force constraints. Horizon N=10, dt=20 ms. |
| `foot_step_planner.py` | Raibert heuristic footstep planner. Computes swing foot landing targets from CoM velocity and generates cubic Bézier swing trajectories. |
| `prioritized_task_execution.py` | Null-space projection WBIC. Tasks are executed in strict priority order (floating base → contact → swing feet → stance).  |
| `robot_model.py` | Pinocchio-based rigid body model wrapper. Exposes mass matrix `M`, Coriolis `C`, gravity `g`, and per-foot Jacobians with correct toe-tip frame IDs. |
| `finalqp.py` | QP layer for distributing desired contact wrenches to joint torques subject to torque limits. |
| `unitree_go2/` | MJCF model assets for the Go2. |


To replicate the repo, make a virtual python environment , then:
```bash
pip install -r requirements.txt
```

---

## Running

```bash

https://github.com/user-attachments/assets/70cca790-8f9a-40c3-8c02-ec8e94840d2f


# Run the full simulation
python main_control_loop.py

```

---

## MPC Formulation

The centroidal MPC uses a **Single Rigid Body (SRB)** model with a 12-D state:

```
x = [roll, pitch, yaw,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz]
```

and 12 decision variables per step (3D ground reaction force per leg). The linearised dynamics are:

```
x[k+1] = A(ψ) x[k] + B(r) f[k] + g_vec
```

where `r` is the foot position relative to the CoM and `ψ` is the yaw. Constraints enforced at each stance step:

- **Friction pyramid:** `|fx|, |fy| ≤ μ fz`, with `μ = 0.6`
- **Unilateral:** `fz ≥ fz_min` (stance), `f = 0` (swing)

Cost weights: position/orientation (Q diagonal), force regularisation (R = 0.1 I).

---

## WBIC / Prioritized Task Execution

The `PrioritizedTaskExecution` class solves a hierarchy of Cartesian tasks using null-space projection:

**Priority order (highest → lowest):**
1. Floating base stability
2. Swing leg tracking
3. Stance leg posture control 


---
###  Final QP Optimization (WBIC)
After the Prioritized Task Execution layer computes the kinematically desired joint accelerations $\ddot{q}_{cmd}$, the `finalqp.py` layer formulates a Quadratic Program to find the optimal ground reaction forces $f_r$. 

This step acts as a bridge between the high-level MPC force commands ($f_{MPC}$) and the low-level task accelerations, ensuring that the physical torques respect the floating-base dynamics and contact constraints. The QP minimizes the deviation from the MPC-planned reaction forces.

