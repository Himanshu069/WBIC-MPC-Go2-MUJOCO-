import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime

class SimulationLogger:
    def __init__(self):
        self.time = []
        self.pos_act = []
        self.pos_des = []
        self.rpy_act = []
        self.rpy_des = []
        self.vel_act = []  
        self.vel_des = []  
        self.torques = []
        self.forces = []
        self.joint_forces = []
        self.foot_z_act = []  
        self.foot_z_des = []  
        self.joint_errors = []

    def log(self, t, pos_act, pos_des, rpy_act, rpy_des, vel_act, vel_des, tau, forces, j_forces, foot_z_act, foot_z_des, j_err):
        """Appends current state data to the logs."""
        self.time.append(t)
        self.pos_act.append(np.array(pos_act).copy())
        self.pos_des.append(np.array(pos_des).copy())
        self.rpy_act.append(np.array(rpy_act).copy())
        self.rpy_des.append(np.array(rpy_des).copy())
        self.vel_act.append(np.array(vel_act).copy())
        self.vel_des.append(np.array(vel_des).copy())
        self.torques.append(np.array(tau).copy())
        self.forces.append(np.array(forces).copy())
        self.joint_forces.append(np.array(j_forces).copy())
        self.foot_z_act.append(np.array(foot_z_act).copy())
        self.foot_z_des.append(np.array(foot_z_des).copy())
        self.joint_errors.append(np.array(j_err).copy())

    def plot(self, base_folder="simulation_plots"):
        """Generates, saves to a unique timestamped folder, and displays analysis plots."""
        t = np.array(self.time)
        if len(t) == 0:
            print("[Plotter] No data logged to plot.")
            return

        # Create a unique directory per run using a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(base_folder, f"run_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        print(f"[Plotter] Saving unique run data to: '{run_folder}/'")

        # Convert lists to numpy arrays
        p_act, p_des = np.array(self.pos_act), np.array(self.pos_des)
        r_act, r_des = np.degrees(np.array(self.rpy_act)), np.degrees(np.array(self.rpy_des))
        v_act, v_des = np.array(self.vel_act), np.array(self.vel_des)
        tau = np.array(self.torques)
        f_grf = np.array(self.forces)
        j_f = np.array(self.joint_forces)
        fz_act, fz_des = np.array(self.foot_z_act), np.array(self.foot_z_des)
        j_err = np.array(self.joint_errors)

        csv_filename = os.path.join(run_folder, "simulation_data.csv")
        leg_names = ['FL', 'FR', 'RL', 'RR']
        joint_names = ['Hip', 'Thigh', 'Calf']

        header = ['Time']
        header += ['Base_Pos_Act_X', 'Base_Pos_Act_Y', 'Base_Pos_Act_Z']
        header += ['Base_Pos_Des_X', 'Base_Pos_Des_Y', 'Base_Pos_Des_Z']
        header += ['Base_RPY_Act_R', 'Base_RPY_Act_P', 'Base_RPY_Act_Y']
        header += ['Base_RPY_Des_R', 'Base_RPY_Des_P', 'Base_RPY_Des_Y']
        header += ['Base_Vel_Act_X', 'Base_Vel_Act_Y', 'Base_Vel_Act_Z']
        header += ['Base_Vel_Des_X', 'Base_Vel_Des_Y', 'Base_Vel_Des_Z']
        for leg in leg_names:
            for joint in joint_names:
                header.append(f'Torque_{leg}_{joint}')
        for leg in leg_names:
            header += [f'GRF_{leg}_X', f'GRF_{leg}_Y', f'GRF_{leg}_Z']
        for leg in leg_names:
            for joint in joint_names:
                header.append(f'Joint_Error_{leg}_{joint}_rad')
        for leg in leg_names:
            header += [f'Foot_Z_Act_{leg}', f'Foot_Z_Des_{leg}']

        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx in range(len(t)):
                row = [t[idx]]
                row += p_act[idx].tolist() + p_des[idx].tolist()
                row += np.array(self.rpy_act)[idx].tolist() + np.array(self.rpy_des)[idx].tolist()
                row += v_act[idx].tolist() + v_des[idx].tolist()
                row += tau[idx].tolist()
                row += f_grf[idx].tolist()
                row += j_err[idx].tolist()
                for leg_i in range(4):
                    row += [fz_act[idx, leg_i], fz_des[idx, leg_i]]
                writer.writerow(row)
        print(f"[Plotter] Raw data spreadsheet exported to: '{csv_filename}'")

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
        # 1. Base Position Tracking
        fig1, axs1 = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
        fig1.suptitle("Base Position Tracking (World Frame)", fontsize=13, fontweight='bold')
        labels_p = ['X (m)', 'Y (m)', 'Z (m)']
        for i in range(3):
            axs1[i].plot(t, p_des[:, i], 'r--', label='Desired', linewidth=1.5)
            axs1[i].plot(t, p_act[:, i], 'b-', label='Actual', linewidth=1.5)
            axs1[i].set_ylabel(labels_p[i])
            axs1[i].legend(loc='upper right')
        axs1[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig1.savefig(os.path.join(run_folder, "1_base_position.png"), dpi=300)

        # 2. Base Orientation Tracking
        fig2, axs2 = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
        fig2.suptitle("Base Orientation Tracking (RPY)", fontsize=13, fontweight='bold')
        labels_r = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
        for i in range(3):
            axs2[i].plot(t, r_des[:, i], 'r--', label='Desired', linewidth=1.5)
            axs2[i].plot(t, r_act[:, i], 'b-', label='Actual', linewidth=1.5)
            axs2[i].set_ylabel(labels_r[i])
            axs2[i].legend(loc='upper right')
        axs2[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig2.savefig(os.path.join(run_folder, "2_base_orientation.png"), dpi=300)

        # 3. NEW: Base Velocity Tracking (Crucial for MPC analysis)
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        fig3.suptitle("Base Linear Velocity Tracking", fontsize=13, fontweight='bold')
        labels_v = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
        for i in range(3):
            axs3[i].plot(t, v_des[:, i], 'r--', label='Target', linewidth=1.5)
            axs3[i].plot(t, v_act[:, i], 'g-', label='Actual', linewidth=1.5)
            axs3[i].set_ylabel(labels_v[i])
            axs3[i].legend(loc='upper right')
        axs3[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig3.savefig(os.path.join(run_folder, "3_base_velocity.png"), dpi=300)

        # 4. Actuator Torques
        fig4, axs4 = plt.subplots(4, 3, figsize=(11, 7), sharex=True)
        fig4.suptitle("Joint Actuator Torques (Nm)", fontsize=13, fontweight='bold')
        for leg_idx in range(4):
            for joint_idx in range(3):
                ax = axs4[leg_idx, joint_idx]
                ax.plot(t, tau[:, leg_idx * 3 + joint_idx], color='purple', linewidth=1.2)
                ax.set_ylabel(f"{leg_names[leg_idx]} {joint_names[joint_idx]}")
                if leg_idx == 3: ax.set_xlabel("Time (s)")
        plt.tight_layout()
        fig4.savefig(os.path.join(run_folder, "4_joint_torques.png"), dpi=300)

        # 5. Ground Reaction Forces (GRFs)
        fig5, axs5 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig5.suptitle("Ground Reaction Forces (World Frame)", fontsize=13, fontweight='bold')
        for leg_idx, leg in enumerate(leg_names):
            ax = axs5[leg_idx]
            start_idx = leg_idx * 3
            ax.plot(t, f_grf[:, start_idx], label='Fx', alpha=0.7)
            ax.plot(t, f_grf[:, start_idx+1], label='Fy', alpha=0.7)
            ax.plot(t, f_grf[:, start_idx+2], label='Fz', linewidth=1.5)
            ax.set_ylabel(f"{leg} Forces (N)")
            ax.legend(loc='upper right', ncol=3)
        axs5[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig5.savefig(os.path.join(run_folder, "5_ground_forces.png"), dpi=300)

        # 6. Joint Bearing Reaction Forces
        fig6, axs6 = plt.subplots(4, 3, figsize=(11, 7), sharex=True)
        fig6.suptitle("Joint Position Tracking Errors (Absolute Deg Delta)", fontsize=13, fontweight='bold')
        
        # Calculate the absolute difference between your target states and actual states
        # Assuming you pass joint tracking data into this array slot now
        for leg_idx in range(4):
            for joint_idx in range(3):
                ax = axs6[leg_idx, joint_idx]
                # Displays the error converted to readable degrees
                ax.plot(t, np.degrees(j_err[:, leg_idx * 3 + joint_idx]), color='crimson', linewidth=1.2)
                ax.set_ylabel(f"{leg_names[leg_idx]} {joint_names[joint_idx]} (deg)")
                if leg_idx == 3: ax.set_xlabel("Time (s)")
        plt.tight_layout()
        fig6.savefig(os.path.join(run_folder, "6_joint_tracking_errors.png"), dpi=300)

        # 7. NEW: Swing/Stance Foot Z-Height Trajectory Tracking
        fig7, axs7 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig7.suptitle("Foot Vertical (Z) Trajectory Tracking (World Frame)", fontsize=13, fontweight='bold')
        for leg_idx, leg in enumerate(leg_names):
            ax = axs7[leg_idx]
            ax.plot(t, fz_des[:, leg_idx], 'r--', label='Planner Target', linewidth=1.5)
            ax.plot(t, fz_act[:, leg_idx], 'b-', label='Actual Pos', linewidth=1.2)
            ax.set_ylabel(f"{leg} Z (m)")
            ax.legend(loc='upper right')
        axs7[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig7.savefig(os.path.join(run_folder, "7_foot_z_tracking.png"), dpi=300)
        
        print(f"[Plotter] All 7 figures saved to '{run_folder}'. Displaying windows...")
        plt.show()