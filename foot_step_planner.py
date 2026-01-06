import numpy as np
from numpy import cos, sin

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