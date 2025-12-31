import numpy as np
from scipy.linalg import block_diag


def MPCConstraints(Ad, Bd, b, x0, param):
    Nx = param['Nx']
    Nu = param['Nu']
    Nt = param['Nt']
    
    # Initialize A matrix
    A = np.zeros((Nx*(Nt+1) + Nu*Nt, Nt*(Nx+Nu) + Nx))
    Noff = Nx*(Nt+1)
    
    # Dynamics constraints
    for k in range(Nt):
        A[k*Nx:(k+1)*Nx, k*Nx:(k+1)*Nx] = -Ad
        A[k*Nx:(k+1)*Nx, (k+1)*Nx:(k+2)*Nx] = np.eye(Nx)
        A[k*Nx:(k+1)*Nx, Noff + k*Nu:Noff + (k+1)*Nu] = -Bd
    
    # Initial state
    A[Nt*Nx:(Nt+1)*Nx, 0:Nx] = np.eye(Nx)
    
    # Input constraints
    for k in range(Nt):
        A[Noff + k*Nu:Noff + (k+1)*Nu, Noff + k*Nu:Noff + (k+1)*Nu] = np.eye(Nu)
    
    # Lower and upper bounds
    bmin = np.zeros(A.shape[0])
    bmin[0:Nt*Nx] = np.tile(b, Nt)
    bmin[Nt*Nx:(Nt+1)*Nx] = x0.flatten()
    
    bmax = bmin.copy()
    bmax[(Nt+1)*Nx:] = np.tile(param['umax'].flatten(), Nt)
    bmin[(Nt+1)*Nx:] = np.tile(param['umin'].flatten(), Nt)
    
    # Optional velocity constraints 
    # Av = np.zeros((int(Nx/2) * Nt, Nt*(Nx + Nu) + Nx))
    # for k in range(1, Nt+1):
    #     Av[(k-1)*6:k*6, Nx*k+6:Nx*(k+1)] = np.eye(6)
    # bvmin = np.tile(-1.5*np.ones(6), Nt)
    # bvmax = -bvmin
    # A = np.vstack([A, Av])
    # bmin = np.hstack([bmin, bvmin])
    # bmax = np.hstack([bmax, bvmax])
    
    return A, bmin, bmax

def eMPCConstraints(xid, p0, x0, param):
    """
    Python translation of MATLAB eMPCConstraints

    min x'Qx + u'Ru
    x_{k+1} = A_k x_k + B_k u_k + b_k
    """

    I = param.I          # inertia matrix (6x6)
    dt = param.dt

    Nx = param.Nx        # typically 12
    Nu = param.Nu        # typically 6
    Nt = param.Nt        # horizon length

    # Total constraint matrix size
    A = np.zeros((Nx * (Nt + 1) + Nu * Nt,
                  Nt * (Nx + Nu) + Nx))

    Noff = Nx * (Nt + 1)

    bmin = np.zeros((A.shape[0], 1))

    for k in range(1, Nt + 1):
        # MATLAB: xi_bar = I * x0(8:13)
        xi_bar = I @ x0[7:13]   # Python is 0-indexed

        G = np.zeros((6, 6))
        G[0:3, 0:3] = skew(xi_bar[0:3])
        G[0:3, 3:6] = skew(xi_bar[3:6])
        G[3:6, 0:3] = skew(xi_bar[3:6])

        H = np.linalg.inv(I) @ (coadjoint(x0[7:13]) @ I + G)

        Ac = np.block([
            [-adjoint_(xid[k-1, :]), -np.eye(6)],
            [np.zeros((6, 6)),        H]
        ])

        Bc = np.vstack([
            np.zeros((6, 6)),
            np.linalg.inv(I)
        ])

        b = -np.linalg.inv(I) @ G @ x0[7:13]
        hc = np.vstack([xid[k-1, :].reshape(-1, 1), b.reshape(-1, 1)])

        # Forward Euler discretization (same as MATLAB)
        Ad = np.eye(Nx) + Ac * dt
        Bd = dt * Bc
        hd = dt * hc

        # State transition constraints
        A[(k-1)*Nx:k*Nx, (k-1)*Nx:k*Nx] = -Ad
        A[(k-1)*Nx:k*Nx, k*Nx:(k+1)*Nx] = np.eye(Nx)
        A[(k-1)*Nx:k*Nx, Noff + (k-1)*Nu:Noff + k*Nu] = -Bd

        bmin[(k-1)*Nx:k*Nx, 0] = hd[:, 0]

    # Initial condition constraint
    A[Nt*Nx:(Nt+1)*Nx, 0:Nx] = np.eye(Nx)

    # Input identity blocks
    for k in range(1, Nt + 1):
        A[Noff + (k-1)*Nu:Noff + k*Nu,
          Noff + (k-1)*Nu:Noff + k*Nu] = np.eye(Nu)

    # Initial state
    bmin[Nt*Nx:(Nt+1)*Nx, 0] = p0.reshape(-1)

    bmax = bmin.copy()
    bmax[(Nt+1)*Nx:, 0] = np.tile(param.umax.reshape(-1), Nt)
    bmin[(Nt+1)*Nx:, 0] = np.tile(param.umin.reshape(-1), Nt)

    return A, bmin, bmax

def eMPCCost(Q, R, P, xid, param):
    """
    Python translation of MATLAB eMPCCost

    min x'Qx + (xi-xid)'Q(xi-xid) + u'Ru
    """

    # Initialize
    M = Q * 0
    q = np.zeros(((param.Nt + 1) * param.Nx + param.Nt * param.Nu, 1))

    # Intermediate stages
    for k in range(1, param.Nt):
        # Transport map
        C = np.eye(12)
        C[6:12, 0:6] = -adjoint_(xid[k-1, :])

        M = block_diag(M, C.T @ Q @ C)

        b = np.zeros((12, 1))
        b[6:12, 0] = xid[k-1, :]

        b = C.T @ Q @ b

        q[(k-1)*param.Nx : k*param.Nx, 0] = -b[:, 0]

    # Terminal cost
    k = param.Nt
    C = np.eye(12)
    C[6:12, 0:6] = -adjoint_(xid[k-1, :])

    b = np.zeros((12, 1))
    b[6:12, 0] = xid[k-1, :]

    b = C.T @ P @ b

    M = block_diag(M, C.T @ P @ C)

    q[(k-1)*param.Nx : k*param.Nx, 0] = -b[:, 0]

    # Control cost blocks
    for _ in range(param.Nt):
        M = block_diag(M, R)

    return M, q