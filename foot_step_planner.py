import numpy as np
from numpy import cos, sin

STEP_HEIGHT = 0.12

class FootStepPlanner:
    def __init__(self, pin_model, stance_time=0.25, k_symmetry=0.03, v_cmd=None, omega_cmd=None):
        self.pin_model = pin_model
        self.stance_time = stance_time
        self.k_symmetry = k_symmetry
        self.v_cmd = np.zeros(3) if v_cmd is None else v_cmd
        self.omega_cmd = np.zeros(3) if omega_cmd is None else omega_cmd

    def compute_next_foot_positions(self):
        """
        Compute next footstep positions in world frame for each leg using Raibert heuristic.
        """
        pm = self.pin_model
        base_pos = pm.pos_com_world
        yaw = pm.current_config.rpy_world()[2]
        h = pm.pos_com_world[2]

        # Rotation around Z
        Rz = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw),  cos(yaw), 0],
            [0, 0, 1]
        ])

        # Hip offsets
        hips = {
            "FL": pm.FL_hip_offset,
            "FR": pm.FR_hip_offset,
            "RL": pm.RL_hip_offset,
            "RR": pm.RR_hip_offset
        }

        # Raibert symmetry term
        v_com = pm.vel_com_world
        psym = self.k_symmetry * (v_com - self.v_cmd) + v_com* (self.stance_time / 2)

        
        omega = np.array([0, 0, self.omega_cmd[2]])  
        pcentrifugal = 0.5 * np.sqrt(h / 9.81) * np.cross(v_com, omega)
        # Compute foot positions
        foot_positions = {}
        for leg, hip_offset in hips.items():
            p_shoulder = base_pos + Rz @ hip_offset
            foot_positions[leg] = p_shoulder + psym + pcentrifugal

        return foot_positions
    
    def swing_foot_target(self, p_lift: np.ndarray, p_land: np.ndarray,
                          phase: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        phase in [0, 1]: 0 = lift-off, 1 = touch-down.
        XY: linear interp between lift and land.
        Z:  raised-cosine arc peaking at STEP_HEIGHT above lift-off Z.
        """
        t_swing = self.stance_time
        dphase = 1.0 / t_swing

        xy = (1 - phase) * p_lift[:2] + phase * p_land[:2]
        z  = p_lift[2] + STEP_HEIGHT * (np.sin(np.pi * phase) ** 2)
        x_des = np.array([xy[0], xy[1], z])

        xy_dot = (p_land[:2] - p_lift[:2]) * dphase
        z_dot = STEP_HEIGHT * np.pi * np.sin(2.0 * np.pi * phase) * dphase
        x_dot_des = np.array([xy_dot[0], xy_dot[1], z_dot])

        xy_ddot = np.zeros(2)
        z_ddot = -STEP_HEIGHT * (np.pi ** 2) * np.cos(2.0 * np.pi * phase) * (dphase ** 2)
        x_ddot_des = np.array([xy_ddot[0], xy_ddot[1], z_ddot])
        
        return x_des, x_dot_des, x_ddot_des