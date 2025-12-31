import numpy as np
import sympy as sp

def rotx(t):
    R = np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t), np.cos(t)]
    ])
    return R

def roty(t):
    R = np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t), 0, np.cos(t)]
    ])
    return R

def rotz(t):
    R = np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ])
    return R

def RPY2Mat(rpy):
    """
    rpy: [roll, pitch, yaw] in radians
    Returns the rotation matrix corresponding to R_z(yaw) * R_y(pitch) * R_x(roll)
    """
    roll, pitch, yaw = rpy
    R = rotz(yaw) @ roty(pitch) @ rotx(roll)
    return R


# Define symbols
x, y, z = sp.symbols('x y z', real=True)      # RPY angles
wx, wy, wz = sp.symbols('wx wy wz', real=True)  # RPY rates

# Rotation matrices
def srotx(t):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(t), -sp.sin(t)],
        [0, sp.sin(t), sp.cos(t)]
    ])

def sroty(t):
    return sp.Matrix([
        [sp.cos(t), 0, sp.sin(t)],
        [0, 1, 0],
        [-sp.sin(t), 0, sp.cos(t)]
    ])

def srotz(t):
    return sp.Matrix([
        [sp.cos(t), -sp.sin(t), 0],
        [sp.sin(t), sp.cos(t), 0],
        [0, 0, 1]
    ])

# RPY rotation matrix
R = srotz(z) * sroty(y) * srotx(x)

# Compute derivative matrix dR
dR = sp.Matrix.zeros(3, 3)
for i in range(3):
    for j in range(3):
        dR[i, j] = sp.Matrix([sp.diff(R[i, j], x),
                              sp.diff(R[i, j], y),
                              sp.diff(R[i, j], z)]).dot(sp.Matrix([wx, wy, wz]))


def dRPY2dw(rpy, drpy):
    """
    Convert RPY rates to angular velocity in body frame.
    
    Parameters:
        rpy: array-like of shape (3,) -> [roll, pitch, yaw] in radians
        drpy: array-like of shape (3,) -> [roll_rate, pitch_rate, yaw_rate]
        
    Returns:
        wb: numpy array of shape (3,) -> angular velocity in body frame
    """
    x, y, z = rpy
    wx, wy, wz = drpy

    wb = np.array([
        wx - wz * np.sin(y),
        wy * np.cos(x) + wz * np.cos(y) * np.sin(x),
        wz * np.cos(x) * np.cos(y) - wy * np.sin(x)
    ])
    
    return wb

# Compute angular velocity matrix in body frame
ww = R.T * dR
ww = sp.simplify(ww)

# Access specific elements
print("ww[2,1] =", ww[2,1])  # MATLAB ww(3,2)
print("ww[0,2] =", ww[0,2])  # MATLAB ww(1,3)
print("ww[1,0] =", ww[1,0])  # MATLAB ww(2,1)
