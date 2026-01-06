import casadi as ca
import numpy as np

def wbic_qp_solver_wbic(A, b, g, Jc, Sf, fr_MPC, q_ddot_cmd, W, n_j, Q1=None, Q2=None):
    """
    WBIC QP solver:
      - Decision variables: delta_fr, delta_f (floating base)
      - Equality constraints: floating base dynamics
      - Inequality constraints: contact (W * fr >= 0)
      - Reaction forces: fr = fr_MPC + delta_fr
      - q̈ = q̈_cmd + [delta_f; 0_joints]
    """
    n_q = q_ddot_cmd.shape[0]        # total DOFs
    n_fb = n_q - n_j                 # floating base DOFs
    n_r = fr_MPC.shape[0]

    # Default weights
    if Q1 is None:
        Q1 = np.eye(n_r)
    if Q2 is None:
        Q2 = np.eye(n_fb) * 0.1

    # Decision variables
    delta_fr = ca.SX.sym('delta_fr', n_r)
    delta_f = ca.SX.sym('delta_f', n_fb)  # floating base relaxation

    # Reaction forces after relaxation
    fr = fr_MPC + delta_fr

    # Full acceleration vector: floating base relaxed, joints fixed
    delta_q_ddot = ca.vertcat(delta_f, ca.SX.zeros(n_j))
    q_ddot = q_ddot_cmd + delta_q_ddot

    # Objective: penalize deviations
    obj = ca.mtimes([delta_fr.T, Q1, delta_fr]) + ca.mtimes([delta_f.T, Q2, delta_f])

    # -----------------------------
    # Constraints
    # -----------------------------
    # 1) Floating-base dynamics (equality)
    dyn_eq = Sf @ (A @ q_ddot + b + g) - Sf @ (Jc.T @ fr)

    # 2) Contact constraints (inequality)
    contact_ineq = W @ fr

    # Concatenate constraints for IPOPT
    g = ca.vertcat(dyn_eq, contact_ineq)

    # Bounds: equality constraints
    lbg = np.zeros(dyn_eq.shape[0])
    ubg = np.zeros(dyn_eq.shape[0])

    # Bounds: inequality constraints
    lbg = np.concatenate([lbg, np.zeros(contact_ineq.shape[0])])
    ubg = np.concatenate([ubg, np.full(contact_ineq.shape[0], np.inf)])

    # Solver setup
    vars = ca.vertcat(delta_fr, delta_f)
    nlp = {'x': vars, 'f': obj, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level':0, 'print_time':0})

    # Initial guess
    x0 = np.zeros(n_r + n_fb)

    # Solve QP
    sol = solver(x0=x0, lbg=lbg, ubg=ubg)
    delta_sol = sol['x'].full().flatten()

    # Extract solution
    delta_fr_opt = delta_sol[:n_r]
    delta_f_opt = delta_sol[n_r:]
    fr_opt = fr_MPC + delta_fr_opt

    return fr_opt, delta_fr_opt, delta_f_opt
