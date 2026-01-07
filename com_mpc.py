import casadi as ca
import numpy as np

# -------------------------------
# Robot Parameters
# -------------------------------
m = 6.921  # base mass [kg]
g = 9.81
I_body = np.diag([0.107027, 0.0980771, 0.0244531])

foot_positions = {
    "FL": np.array([0.1934, 0.0465, -0.213]),
    "FR": np.array([0.1934, -0.0465, -0.213]),
    "RL": np.array([-0.1934, 0.0465, -0.213]),
    "RR": np.array([-0.1934, -0.0465, -0.213])
}
foot_names = ["FL", "FR", "RL", "RR"]
n_legs = 4

# MPC parameters
N = 10      # horizon
dt = 0.02
Q = np.diag([50,50,50,100,100,100,1,1,1,10,10,10])
R = 0.1*np.eye(3*n_legs)
mu = 0.6
fz_min = 1.0  # minimal normal force

# -------------------------------
# Utility functions
# -------------------------------
def cross_mat(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0)
    )

def Rz(psi):
    return ca.vertcat(
        ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
        ca.horzcat(ca.sin(psi), ca.cos(psi), 0),
        ca.horzcat(0, 0, 1)
    )

# -------------------------------
# Centroidal MPC
# -------------------------------
def centroidal_mpc(x0, x_ref_traj, contact_schedule):
    X = ca.SX.sym('X', 12, N+1)
    F = ca.SX.sym('F', 3*n_legs, N)

    g_constr = []
    lbg = []
    ubg = []

    cost = 0

    # Initial state
    g_constr.append(X[:,0] - x0)
    lbg += [0]*12
    ubg += [0]*12

    for k in range(N):
        psi_k = x_ref_traj[k,2]
        Rz_k = Rz(psi_k)

        # Dynamics matrices (linearized)
        A = ca.SX.eye(12)
        A[0:3,6:9] = Rz_k*dt
        A[3:6,9:12] = dt*ca.SX.eye(3)

        B = ca.SX.zeros(12, 3*n_legs)
        for i, foot in enumerate(foot_names):
            r = foot_positions[foot]
            B[6:9, 3*i:3*i+3] = ca.mtimes(ca.inv(I_body), cross_mat(r)) * dt
            B[9:12, 3*i:3*i+3] = dt/m * ca.SX.eye(3)

        g_vec = ca.vertcat(ca.SX.zeros(9), ca.SX([0,0,-g]))*dt

        x_next = ca.mtimes(A, X[:,k]) + ca.mtimes(B, F[:,k]) + g_vec
        g_constr.append(X[:,k+1] - x_next)
        lbg += [0]*12
        ubg += [0]*12

        # Cost
        x_err = X[:,k+1] - x_ref_traj[k,:]
        cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([F[:,k].T, R, F[:,k]])

        # Friction & unilateral constraints
        for i, foot in enumerate(foot_names):
            fx = F[3*i, k]
            fy = F[3*i+1, k]
            fz = F[3*i+2, k]

            if contact_schedule[k,i] == 1:  # stance
                # Friction pyramid
                # |fx| <= mu fz
                g_constr.append(fx - mu*fz)
                lbg.append(-ca.inf)
                ubg.append(0)

                g_constr.append(-fx - mu*fz)
                lbg.append(-ca.inf)
                ubg.append(0)

                # |fy| <= mu fz
                g_constr.append(fy - mu*fz)
                lbg.append(-ca.inf)
                ubg.append(0)

                g_constr.append(-fy - mu*fz)
                lbg.append(-ca.inf)
                ubg.append(0)

                # fz > 0  (numerically: fz >= 0 or small epsilon)
                g_constr.append(fz)
                lbg.append(fz_min)          # or fz_min
                ubg.append(ca.inf)
            else:
                    g_constr.append(fx)
                    g_constr.append(fy)
                    g_constr.append(fz)
                    lbg += [0, 0, 0]
                    ubg += [0, 0, 0]

    g_constr = ca.vertcat(*g_constr)
    vars = ca.vertcat(ca.reshape(X, -1,1), ca.reshape(F,-1,1))

    # Solver
    nlp = {'x': vars, 'f': cost, 'g': g_constr}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level':0, 'print_time':0})

    # Initial guess
    X_guess = np.tile(x0, (N+1,1)).T
    F_guess = np.zeros((3*n_legs, N))
    # Set vertical forces to support weight
    for k in range(N):
        for i in range(n_legs):
            F_guess[3*i+2, k] = m*g/n_legs

    x0_guess = np.vstack([X_guess.reshape(-1,1), F_guess.reshape(-1,1)])

    assert len(lbg) == g_constr.shape[0]
    assert len(ubg) == g_constr.shape[0]
    sol = solver(
        x0=x0_guess,
        lbg=np.array(lbg),
        ubg=np.array(ubg)
    )
    X_opt = np.array(sol['x'][0:12*(N+1)]).reshape(12, N+1)
    F_opt = np.array(sol['x'][12*(N+1):]).reshape(3*n_legs, N)

    return X_opt, F_opt
