import numpy as np
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===============================
# Placeholder imports (replace with your implementations)
# ===============================
from EMPC.control_EMPC import eMPC_casadi  # your proposed MPC
from EMPC.Lie_Group import quat_to_rotm, skew, SE3Dyn

# ===============================
# Param container
# ===============================
class Param:
    pass

# ===============================
# MPC Simulation (Proposed MPC)
# ===============================
def sim_mpc_proposed(q0, p0, w0, v0, dt, Nsim, param):
    X_ref = param.X_ref
    xi_ref = param.xi_ref

    # Cost matrices (same as MATLAB mode==1)
    param.P = np.diag([
        1, 1, 1,
        10, 10, 10,
        1, 1, 1,
        1, 1, 1
    ]) * 10
    param.Q = np.diag([
        1, 1, 1,
        10, 10, 10,
        1, 1, 1,
        1, 1, 1
    ]) * 1
    param.R = np.eye(6) * 1e-5

    # Initial state
    x0 = np.hstack([q0, p0, w0, v0])

    X_log = [x0.copy()]
    U_log = []
    Err_log = []
    Err2_log = []

    for i in range(Nsim - param.Nt):
        X_ref_rt = X_ref[i]
        xi_ref_rt = xi_ref[i:i + param.Nt]

        # MPC control
        u = eMPC_casadi(x0, X_ref_rt, xi_ref_rt, param)
        U_log.append(u)

        # True SE(3) dynamics
        def dyn(t, x):
            return SE3Dyn(t, x, u, param.I)

        sol = solve_ivp(
            dyn,
            (0.0, dt),
            x0,
            method="RK45",
            t_eval=[dt]
        )
        x0 = sol.y[:, -1]

        # Construct poses
        Xd = np.block([
            [quat_to_rotm(X_ref_rt[0:4]), X_ref_rt[4:7].reshape(3, 1)],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ])
        X0 = np.block([
            [quat_to_rotm(x0[0:4]), x0[4:7].reshape(3, 1)],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ])

        # Left-invariant error
        Xerr = logm(np.linalg.inv(X0) @ Xd)
        XXerr = np.linalg.inv(X0) @ Xd

        err_vec = np.zeros(6)
        err_vec[0:3] = np.array([Xerr[2,1], Xerr[0,2], Xerr[1,0]])
        err_vec[3:6] = Xerr[0:3,3]

        rot_err = np.sqrt(np.sum(logm(XXerr[0:3,0:3])**2)/2)
        pos_err = XXerr[0:3,3]

        Err_log.append(err_vec)
        Err2_log.append(np.hstack([rot_err, pos_err]))
        X_log.append(x0.copy())

    logger = {
        "X": np.array(X_log).T,
        "U": np.array(U_log).T,
        "Err": np.array(Err_log).T,
        "Err2": np.array(Err2_log)
    }
    return logger

# ===============================
# Plotting results
# ===============================
def plot_results(logger, dt):
    t = np.arange(logger["Err2"].shape[0]) * dt

    # Orientation error
    plt.figure()
    plt.plot(t, logger["Err2"][:,0])
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation error [rad]")
    plt.title("Orientation Error")
    plt.grid()

    # Position error
    plt.figure()
    plt.plot(t, logger["Err2"][:,1], label="x")
    plt.plot(t, logger["Err2"][:,2], label="y")
    plt.plot(t, logger["Err2"][:,3], label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("Position Error")
    plt.legend()
    plt.grid()

    # Control inputs
    plt.figure()
    for i in range(6):
        plt.plot(logger["U"][i,:], label=f"u{i+1}")
    plt.xlabel("Time step")
    plt.ylabel("Control input")
    plt.title("MPC Control Inputs")
    plt.legend()
    plt.grid()

    plt.show()

# ===============================
# Main script
# ===============================
def main():
    # Simulation parameters
    dt = 0.025
    Nsim = int(np.ceil(6.0 / dt))

    param = Param()
    param.Nx = 12
    param.Nu = 6
    param.Nt = 20
    param.dt = dt
    param.umin = -4000*np.ones(6)
    param.umax =  4000*np.ones(6)

    # Inertia (6x6 spatial inertia)
    Ib = np.diag([1.0, 2.0, 3.0])
    M  = np.eye(3)
    param.I = np.block([
        [Ib, np.zeros((3,3))],
        [np.zeros((3,3)), M]
    ])

    # Initial state
    q0 = np.array([1.0,0.0,0.0,0.0])
    p0 = np.zeros(3)
    w0 = np.zeros(3)
    v0 = np.zeros(3)

    # Reference trajectory
    X = np.eye(4)
    param.X_ref = []
    param.xi_ref = []

    w_ref = np.array([0.0,0.0,1.0])
    v_ref = np.array([2.0,0.0,0.2])
    xid_ref = np.hstack([w_ref,v_ref])

    param.X_ref.append(np.hstack([q0,p0]))
    param.xi_ref.append(xid_ref)

    for _ in range(Nsim):
        Xi = np.block([
            [skew(w_ref), v_ref.reshape(3,1)],
            [np.zeros((1,4))]
        ])
        X = X @ expm(Xi*dt)
        R = X[:3,:3]
        p = X[:3,3]

        qw = np.sqrt(1 + np.trace(R))/2
        qx = (R[2,1]-R[1,2])/(4*qw)
        qy = (R[0,2]-R[2,0])/(4*qw)
        qz = (R[1,0]-R[0,1])/(4*qw)

        param.X_ref.append(np.array([qw,qx,qy,qz,*p]))
        param.xi_ref.append(xid_ref)

    param.X_ref = np.array(param.X_ref)
    param.xi_ref = np.array(param.xi_ref)

    # Run simulation
    logger = sim_mpc_proposed(q0,p0,w0,v0,dt,Nsim,param)

    # Plot results
    plot_results(logger, dt)

if __name__ == "__main__":
    main()
