
import numpy as np
from scipy.linalg import logm

def skew(p):
    return np.array([
        [0,     -p[2],  p[1]],
        [p[2],   0,    -p[0]],
        [-p[1],  p[0],  0]
    ])

def quat_to_rotm(q):
    """
    q = [qw, qx, qy, qz]  (scalar-first)
    """
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])

def adjoint_mat(x):
    R = x[0:3, 0:3]
    p = x[0:3, 3]

    Adx = np.block([
        [R,               np.zeros((3, 3))],
        [skew(p) @ R,     R]
    ])

    return Adx

def adjoint(x):
    """
    x ∈ R^6 = [w; v]
    returns ad_x ∈ R^{6x6}
    """
    w = x[0:3]
    v = x[3:6]

    adx = np.block([
        [skew(w), np.zeros((3, 3))],
        [skew(v), skew(w)]
    ])

    return adx

def coadjoint(x):
    """
    x: 6x1 vector [w; v]
    Returns the coadjoint (transpose of ad)
    """
    return adjoint(x).T


def get_error(X, X0, xi, xi0):
    """
    Compute the SE(3) error between two poses and the twist error.

    Parameters:
    -----------
    X : np.ndarray, 4x4
        Current pose (homogeneous transformation)
    X0 : np.ndarray, 4x4
        Reference pose
    xi : np.ndarray, 6x1
        Current twist
    xi0 : np.ndarray, 6x1
        Reference twist

    Returns:
    --------
    eX : np.ndarray, 4x4
        Pose error (X^-1 * X0)
    ex : np.ndarray, 4x4
        Logarithm of pose error (matrix logarithm)
    exi : np.ndarray, 6x1
        Twist error (xi - xi0)
    """
    eX = np.linalg.inv(X) @ X0        # X^{-1} * X0
    ex = logm(eX)                     # matrix logarithm
    exi = xi - xi0                     # twist error
    return eX, ex, exi


def Jac_SO3_L(xi):
    t = np.linalg.norm(xi)
    I = np.eye(3)
    xi_hat = skew(xi)

    J = (
        I
        + (1 - np.cos(t)) / t**2 * xi_hat
        + (t - np.sin(t)) / t**3 * (xi_hat @ xi_hat)
    )
    return J

def Jac_SE3_L(xi, eps=1e-8):
    """
    Left Jacobian of SE(3) for a twist vector xi = [omega, v]
    xi: 6x1 vector (first 3: rotation, last 3: translation)
    """
    omega = xi[:3]
    v = xi[3:]
    t = np.linalg.norm(omega)

    X = np.block([
        [skew(omega), skew(v)],
        [np.zeros((3, 3)), skew(omega)]
    ])

    I6 = np.eye(6)

    # if t < eps:
    #     return I6 + 0.5 * X + (1/6) * X @ X + (1/24) * X @ X @ X + (1/120) * X @ X @ X @ X

    J = (
        I6
        + (4 - t * np.sin(t) - 4 * np.cos(t)) / (2 * t**2) * X
        + (4 * t - 5 * np.sin(t) + t * np.cos(t)) / (2 * t**3) * X @ X
        + (2 - t * np.sin(t) - 2 * np.cos(t)) / (2 * t**4) * X @ X @ X
        + (2 * t - 3 * np.sin(t) + t * np.cos(t)) / (t**5) * X @ X @ X @ X
    )

    J[3:6, 0:3] = J[0:3, 3:6]
    J[0:3, 3:6] = np.zeros((3, 3))

    return J

def SE3_discrete(x, param):
    """
    x: (12,) state vector
       [phi, theta, psi, x, y, z, wx, wy, wz, vx, vy, vz]
    param.dt: timestep
    param.J: 6x6 inertia matrix
    """

    r = x[0:3]      # orientation (Euler)
    p = x[3:6]      # position
    w = x[6:9]      # angular velocity
    v = x[9:12]     # linear velocity

    dt = param.dt
    J = param.J    # shape (6,6)

    I6 = np.eye(6)
    Z6 = np.zeros((6, 6))

    Ad = np.block([
        [I6, dt * I6],
        [Z6, I6]
    ])

    J_inv = np.linalg.inv(J)

    Bd = np.vstack([
        (dt ** 2) * J_inv,
        dt * J_inv
    ])

    b = 0.5 * (dt ** 2)

    return Ad, Bd, b

def SE3Dyn(t, x, u, I):
    """
    x: (13,)
    u: (6,)
    I: (6,6) spatial inertia
    """

    q = x[0:4]
    p = x[4:7]
    w = x[7:10]
    v = x[10:13]

    # Quaternion kinematics
    Omega = np.array([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0]
    ])

    dQuat = 0.5 * Omega @ q

    # Position kinematics
    R = quat_to_rotm(q)
    dp = R @ v

    dx = np.concatenate([dQuat, dp])

    xi = np.concatenate([w, v])

    d2x = np.linalg.inv(I) @ (
        coadjoint(xi) @ I @ xi + u
    )

    dxdt = np.concatenate([dx, d2x])
    return dxdt

