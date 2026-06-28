import numpy as np
from numpy import cos, sin

STEP_HEIGHT = 0.08

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
            target_pos = p_shoulder + psym + pcentrifugal
            foot_positions[leg] = target_pos
            foot_positions[leg][2] = 0.0 
        return foot_positions
    
    def swing_foot_target(self, p_lift: np.ndarray, p_land: np.ndarray,
                          phase: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        phase in [0, 1]: 0 = lift-off, 1 = touch-down.
        """
        s = np.clip(phase, 0.0, 1.0)

        t_swing = self.stance_time
        dphase = 1.0 / t_swing
        dphase2 = dphase ** 2

        c       = 10 * (s**3) - 15 * (s**4) + 6 * (s**5)
        dc_ds   = 30 * (s**2) - 60 * (s**3) + 30 * (s**4)
        d2c_ds2 = 60 * s      - 180 * (s**2) + 120 * (s**3)
        
        xy_diff = p_land[:2] - p_lift[:2]
        
        xy = p_lift[:2] + xy_diff * c
        xy_dot = xy_diff * dc_ds * dphase
        xy_ddot = xy_diff * d2c_ds2 * dphase2

        z_base      =  p_lift[2] + (p_land[2] - p_lift[2]) * c   # uses same quintic c
        z_base_dot  = (p_land[2] - p_lift[2]) * dc_ds  * dphase
        z_base_ddot = (p_land[2] - p_lift[2]) * d2c_ds2 * dphase2

        arc        =  4.0 * STEP_HEIGHT * s * (1.0 - s)
        darc_ds    =  4.0 * STEP_HEIGHT * (1.0 - 2.0 * s)
        d2arc_ds2  = -8.0 * STEP_HEIGHT

        z      = z_base      + arc
        z_dot  = z_base_dot  + darc_ds   * dphase
        z_ddot = z_base_ddot + d2arc_ds2 * dphase2

        x_des      = np.array([xy[0],      xy[1],      z     ])
        x_dot_des  = np.array([xy_dot[0],  xy_dot[1],  z_dot ])
        x_ddot_des = np.array([xy_ddot[0], xy_ddot[1], z_ddot])

        
        return x_des, x_dot_des, x_ddot_des