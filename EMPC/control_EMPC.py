import numpy as np
import casadi as ca
from scipy.linalg import logm
from EMPC.Lie_Group import quat_to_rotm, skew, adjoint, coadjoint

def eMPC_casadi(x0, xd, xid, param):
    """
    Extended MPC for SE(3) dynamics using CasADi.
    Returns the first control input as a numpy array.
    """

    Nx = param.Nx
    Nu = param.Nu
    Nt = param.Nt
    dt = param.dt
    I = param.I
    Q = param.Q
    R = param.R
    P = param.P

    # --- Compute left-invariant error at current state ---
    Xd_SE3 = np.block([
        [quat_to_rotm(xd[0:4]), xd[4:7].reshape(3, 1)],
        [np.zeros((1, 3)), 1]
    ])
    X0_SE3 = np.block([
        [quat_to_rotm(x0[0:4]), x0[4:7].reshape(3, 1)],
        [np.zeros((1, 3)), 1]
    ])
    Xerr_SE3 = logm(np.linalg.inv(X0_SE3) @ Xd_SE3)

    # error vector: rotation (vee of skew) + position
    x_init = np.zeros(Nx)
    x_init[0:3] = [Xerr_SE3[2,1], Xerr_SE3[0,2], Xerr_SE3[1,0]]
    x_init[3:6] = Xerr_SE3[0:3,3]
    x_init[6:] = x0[7:]  # angular + linear velocity

    # --- CasADi variables ---
    X = ca.SX.sym('X', Nx, Nt+1)
    U = ca.SX.sym('U', Nu, Nt)

    cost = 0
    g = []

    for k in range(Nt):
        # Linearized discrete dynamics (Euler approx)
        xi_k = X[:, k]
        u_k = U[:, k]

        # Dynamics linearization: dx/dt = f(x) + Bu ~ dx = dt*(f + Bu)
        # Use identity for simplification; more complex dynamics can replace this
        Ad = ca.DM.eye(Nx)
        Bd = ca.DM.zeros(Nx, Nu)
        Bd[6:12, :] = dt * ca.DM.eye(6)  # angular & linear velocity to accelerations

        x_next = Ad @ xi_k + Bd @ u_k
        g.append(X[:, k+1] - x_next)

        # Cost
        cost += ca.mtimes([xi_k.T, Q, xi_k]) + ca.mtimes([u_k.T, R, u_k])

    # Terminal cost
    cost += ca.mtimes([X[:, Nt].T, P, X[:, Nt]])

    # Flatten constraints
    g = ca.vertcat(*g)

    # --- NLP ---
    nlp_prob = {
        'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
        'f': cost,
        'g': g
    }

    # Solver options
    opts = {'ipopt.print_level':0, 'print_time':0}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # Initial guess
    x0_guess = np.tile(x_init, Nt+1)  # flatten Nx*(Nt+1)
    u0_guess = np.zeros(Nu*Nt)
    x0_all = np.hstack([x0_guess, u0_guess])
    lbx = -ca.inf * np.ones_like(x0_all)
    ubx = ca.inf * np.ones_like(x0_all)
    lbg = np.zeros(g.shape)
    ubg = np.zeros(g.shape)

    sol = solver(x0=x0_all, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    U_opt = sol['x'][-Nu*Nt:].full().reshape(Nu, Nt)
    return U_opt[:, 0]
