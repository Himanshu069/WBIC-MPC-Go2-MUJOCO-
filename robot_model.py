from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy import cos, sin
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from dataclasses import dataclass, field
from typing import List


@dataclass
class ConfigurationState:
    # Base pose
    base_pos: np.ndarray
    base_quat: np.ndarray

    # Joint angles (FL, FR, RL, RR)
    FL_q: np.ndarray
    FR_q: np.ndarray
    RL_q: np.ndarray
    RR_q: np.ndarray

    # Base velocities
    base_v: np.ndarray
    base_w: np.ndarray

    # Joint velocities
    FL_dq: np.ndarray
    FR_dq: np.ndarray
    RL_dq: np.ndarray
    RR_dq: np.ndarray

    @classmethod
    def create_default(cls):
        return cls(
            base_pos=np.array([0.0, 0.0, 0.27]),
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            FL_q=np.array([0.0, 0.9, -1.8]),
            FR_q=np.array([0.0, 0.9, -1.8]),
            RL_q=np.array([0.0, 0.9, -1.8]),
            RR_q=np.array([0.0, 0.9, -1.8]),
            base_v=np.zeros(3),
            base_w=np.zeros(3),
            FL_dq=np.zeros(3),
            FR_dq=np.zeros(3),
            RL_dq=np.zeros(3),
            RR_dq=np.zeros(3)
        )

    # --- packing / unpacking ---
    def q(self) -> np.ndarray: # q is 19 *1
        return np.concatenate([
        self.base_pos,
        self.base_quat,
        self.FL_q, self.FR_q, self.RL_q, self.RR_q
        ])


    def dq(self) -> np.ndarray: #dq is 18 *1
        return np.concatenate([
        self.base_v, self.base_w,
        self.FL_dq, self.FR_dq, self.RL_dq, self.RR_dq
        ])


    def update_q(self, q: np.ndarray):
        self.base_pos = q[0:3]
        self.base_quat = q[3:7]
        self.FL_q, self.FR_q = q[7:10], q[10:13]
        self.RL_q, self.RR_q = q[13:16], q[16:19]


    def update_dq(self, v: np.ndarray):
        self.base_v = v[0:3]
        self.base_w = v[3:6]
        self.FL_dq, self.FR_dq = v[6:9], v[9:12]
        self.RL_dq, self.RR_dq = v[12:15], v[15:18]


    def rpy_world(self) -> np.ndarray:
        q = pin.Quaternion(self.base_quat)
        R = q.toRotationMatrix()
        roll, pitch, yaw = np.array(pin.rpy.matrixToRpy(R)).reshape(3,)


        if not hasattr(self, "_yaw_init"):
            self._yaw_init = True
            self._yaw_prev = yaw
            self._yaw_cont = yaw
        else:
            dy = (yaw - self._yaw_prev + np.pi) % (2 * np.pi) - np.pi
            self._yaw_cont += dy
            self._yaw_prev = yaw


        return np.array([roll, pitch, self._yaw_cont])


    def set_rpy(self, roll: float, pitch: float, yaw: float):
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        self.base_quat = np.array([
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy
        ])
class PinModel:
    MJCF_PATH = Path("unitree_go2") / "go2.xml"

    def __init__(self):
        """Initialize robot model"""
        # Load robot from MJCF
        robot = RobotWrapper.BuildFromMJCF(
            str(self.MJCF_PATH),
            root_joint=pin.JointModelFreeFlyer()
        )

        # Core models
        self.model = robot.model
        self.vmodel = robot.visual_model
        self.cmodel = robot.collision_model
        self.data = self.model.createData()

        # Initial configuration
        self.current_config = ConfigurationState.create_default()
        self.q_init = self.current_config.q()
        self.dq_init = self.current_config.dq()

        # Forward kinematics / frame placements at q_init
        pin.forwardKinematics(self.model, self.data, self.q_init)
        pin.updateFramePlacements(self.model, self.data)

        # Get frame IDs - using calf bodies as foot frames
        self.base_id = self.model.getFrameId("base")

        # Foot frames - using calf bodies (last link in each leg)
        self.FL_foot_id = self.model.getFrameId("FL_calf")
        self.FR_foot_id = self.model.getFrameId("FR_calf")
        self.RL_foot_id = self.model.getFrameId("RL_calf")
        self.RR_foot_id = self.model.getFrameId("RR_calf")

        # Hip frames - using thigh bodies
        self.FL_hip_id = self.model.getFrameId("FL_thigh")
        self.FR_hip_id = self.model.getFrameId("FR_thigh")
        self.RL_hip_id = self.model.getFrameId("RL_thigh")
        self.RR_hip_id = self.model.getFrameId("RR_thigh")

        # Compute hip offsets relative to base
        oMb = self.data.oMf[self.base_id]
        oMh1 = self.data.oMf[self.FL_hip_id]
        oMh2 = self.data.oMf[self.FR_hip_id]
        oMh3 = self.data.oMf[self.RL_hip_id]
        oMh4 = self.data.oMf[self.RR_hip_id]

        bMh1 = oMb.actInv(oMh1)
        bMh2 = oMb.actInv(oMh2)
        bMh3 = oMb.actInv(oMh3)
        bMh4 = oMb.actInv(oMh4)

        self.FL_hip_offset = bMh1.translation.copy()
        self.FR_hip_offset = bMh2.translation.copy()
        self.RL_hip_offset = bMh3.translation.copy()
        self.RR_hip_offset = bMh4.translation.copy()

        # Model state placeholders (initialized in update_model)
        self.oMb = None
        self.oMf1 = None
        self.oMf2 = None
        self.oMf3 = None
        self.oMf4 = None
        self.pos_com_world = None
        self.vel_com_world = None
        self.R_body_to_world = None
        self.R_world_to_body = None
        self.R_z = None

        # Desired trajectories (optional, can be populated later)
        self.x_pos_des_world = []
        self.y_pos_des_world = []
        self.x_vel_des_world = []
        self.y_vel_des_world = []
        self.yaw_rate_des_world = []

        # Initialize model state
        self.update_model(self.q_init, self.dq_init)

    def get_hip_offset(self, leg: str):
        name = f"{leg.upper()}_hip_offset"
        return getattr(self, name)
    
    def compute_com_x_vec(self):
        """Compute 12-DOF centroidal state vector"""
        pos_com_world = self.pos_com_world
        rpy_com_world = self.current_config.rpy_world()
        vel_com_world = self.vel_com_world
        rpy_rate_body = self.current_config.base_w
        
        R = self.R_body_to_world
        omega_world = R @ rpy_rate_body

        x_vec = np.concatenate([pos_com_world, rpy_com_world, 
                                vel_com_world, omega_world])
        
        x_vec = x_vec.reshape(-1, 1)

        return x_vec

    def update_model(self, q, dq):
        """Update model with new configuration and velocities"""
        self.current_config.update_q(q)
        self.current_config.update_dq(dq)
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data) 
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)

        self.oMb = self.data.oMf[self.base_id]
        self.oMf1 = self.data.oMf[self.FL_foot_id]
        self.oMf2 = self.data.oMf[self.FR_foot_id]
        self.oMf3 = self.data.oMf[self.RL_foot_id]
        self.oMf4 = self.data.oMf[self.RR_foot_id]
        self.pos_com_world = self.data.com[0]
        self.vel_com_world = self.data.vcom[0]

        yaw = self.current_config.rpy_world()[2]
        R_bw = np.array(self.oMb.rotation)

        self.R_body_to_world = R_bw
        self.R_world_to_body = R_bw.T

        self.R_z = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw),  cos(yaw), 0],
            [0,             0,     1]
        ])

    def update_model_simplified(self, q, dq):
        """Update model with simplified state (roll, pitch, yaw instead of quaternion)"""
        roll = q[3]
        pitch = q[4]
        yaw = q[5]

        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy

        q_full = np.concatenate([
            q[0:3],                # base position
            [qx, qy, qz, qw],      # base quaternion
            np.zeros(12)           # 12 leg joint angles
        ])

        dq_full = np.concatenate([
            dq[0:6],              
            np.zeros(12)
        ])

        self.update_model(q_full, dq_full)

    def get_foot_placement_in_world(self):
        """Get foot positions in world frame"""
        FL_placement = self.oMf1.translation.copy()
        FR_placement = self.oMf2.translation.copy()
        RL_placement = self.oMf3.translation.copy()
        RR_placement = self.oMf4.translation.copy()

        return FL_placement, FR_placement, RL_placement, RR_placement
    
    def get_foot_lever_world(self):
        """Get foot positions relative to CoM in world frame"""
        pos_com_world = self.pos_com_world    
        FL_placement = self.oMf1.translation - pos_com_world
        FR_placement = self.oMf2.translation - pos_com_world
        RL_placement = self.oMf3.translation - pos_com_world
        RR_placement = self.oMf4.translation - pos_com_world

        return FL_placement, FR_placement, RL_placement, RR_placement
    
    def get_single_foot_state_in_world(self, leg: str):
        """Get position and velocity of a single foot in world frame"""
        foot_id = getattr(self, f"{leg}_foot_id")

        # Position in world (assumes updateFramePlacements already called)
        oMf = self.data.oMf[foot_id]
        foot_pos_world = oMf.translation.copy()  # (3,)

        # 6D spatial velocity in LOCAL_WORLD_ALIGNED (axes = world)
        v6 = pin.getFrameVelocity(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        foot_vel_world = np.array(v6.linear).copy()  # (3,)

        return foot_pos_world, foot_vel_world
    
    def compute_3x3_foot_Jacobian_world(self, leg: str):
        """Compute 3x3 foot Jacobian (linear part only) in world frame for leg joints"""
        foot_id = getattr(self, f"{leg}_foot_id")
        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos_world = J_world[0:3, :]

        # Get joint IDs for this leg (hip, thigh, calf)
        joint_ids = [
            self.model.getJointId(f"{leg}_hip_joint"), 
            self.model.getJointId(f"{leg}_thigh_joint"), 
            self.model.getJointId(f"{leg}_calf_joint")
        ]

        vcols = [self.model.joints[jid].idx_v for jid in joint_ids]
        J_leg_pos_world = J_pos_world[:, vcols] 

        return J_leg_pos_world

    def compute_3x3_foot_Jacobian_body(self, leg: str):
        """Compute 3x3 foot Jacobian in body frame for leg joints"""
        foot_id = getattr(self, f"{leg}_foot_id")

        # 6xnv Jacobian, expressed in WORLD
        J_world = pin.getFrameJacobian(
            self.model, self.data, foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_pos_world = J_world[0:3, :]

        # Base placement in world: oMb
        oMb = self.data.oMf[self.base_id]
        R_wb = oMb.rotation  # R_WB

        # Rotate Jacobian into BODY (base) frame
        J_pos_body = R_wb.T @ J_pos_world

        # Pick the 3 leg joints
        joint_ids = [
            self.model.getJointId(f"{leg}_hip_joint"),
            self.model.getJointId(f"{leg}_thigh_joint"),
            self.model.getJointId(f"{leg}_calf_joint"),
        ]
        vcols = [self.model.joints[jid].idx_v for jid in joint_ids]
        J_leg_pos_body = J_pos_body[:, vcols]

        return J_leg_pos_body
    
    def compute_Jdot_dq_world(self, leg: str):
        """Compute Jdot * dq for foot in world frame"""
        foot_id = getattr(self, f"{leg}_foot_id")

        Jdot = pin.getFrameJacobianTimeVariation(
            self.model, self.data, foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        Jdot_dq = Jdot[0:3, :] @ self.current_config.dq()
        return np.asarray(Jdot_dq).reshape(3,)
    
    def compute_full_foot_Jacobian_world(self, leg: str):
        """Compute full (3 x nv) foot Jacobian in world frame"""
        foot_id = getattr(self, f"{leg}_foot_id")
        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos_world = J_world[0:3, :]
        return J_pos_world
    
    def compute_dynamics_terms(self):
        """Get gravity, Coriolis, and mass matrix"""
        g = self.data.g  # gravity torque term
        C = self.data.C  # Coriolis matrix
        M = self.data.M  # joint-space inertia matrix
        return g, C, M

    def run_simulation(self, u_vec):
        """Run forward simulation with control inputs"""
        N_input = u_vec.shape[1]
        assert N_input == self.dynamics_N, f"Expected {N_input=} to equal {self.dynamics_N=}"

        x_traj = np.zeros((12, N_input+1))
        x_init = self.compute_com_x_vec()
        x_traj[:, [0]] = x_init

        for i in range(N_input):
            u_i = u_vec[:, i].reshape(-1, 1)
            x_traj[:, i+1] = (self.Ad @ x_traj[:, [i]] + self.Bd[i] @ u_i + self.gd).flatten()

        return x_init, x_traj