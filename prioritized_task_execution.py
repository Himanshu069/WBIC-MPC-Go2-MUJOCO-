import numpy as np

class PrioritizedTaskExecution:
    def __init__(self, n_joints):
        """
        n_joints: number of joints in the robot
        """
        self.n_joints = n_joints

    @staticmethod
    def pseudo_inverse(J, method='svd', A=None):
        """
        Compute pseudo-inverse.
        method: 'svd' or 'dynamically_consistent'
        If dynamically_consistent, A should be joint-space inertia matrix
        """
        if method == 'svd':
            return np.linalg.pinv(J)
        elif method == 'dynamically_consistent':
            if A is None:
                raise ValueError("A (mass/inertia matrix) required for dynamically consistent pseudo-inverse")
            A_inv = np.linalg.inv(A)
            return A_inv @ J.T @ np.linalg.inv(J @ A_inv @ J.T)
        else:
            raise ValueError("Invalid method")

    def execute(self, tasks, q_curr, q_dot_curr, A=None, Jc=None):
        """
        Compute desired joint position, velocity, acceleration for prioritized tasks.

        tasks: list of dicts, each with keys:
            - 'J': Jacobian of the task (m_i x n_joints)
            - 'x_des': desired task-space position (m_i,)
            - 'x_dot_des': desired velocity (m_i,)
            - 'x_ddot_des': desired acceleration (m_i,)
            - 'Kp': proportional gain
            - 'Kd': derivative gain
        q_curr: current joint positions (n_joints,)
        q_dot_curr: current joint velocities (n_joints,)
        A: joint-space inertia matrix (n_joints x n_joints), optional for dyn-consistent pseudo-inverse

        Returns:
            delta_q: change in joint positions
            q_dot_cmd: desired joint velocities
            q_ddot_cmd: desired joint accelerations
        """
        n = self.n_joints
        delta_q_prev = np.zeros(n)
        q_dot_prev = np.zeros(n)
        q_ddot_prev = np.zeros(n)

        if Jc is not None:
            Jc_pinv = self.pseudo_inverse(Jc, method='svd',)
            N_prev = np.eye(n) - Jc_pinv @ Jc
        else:
            N_prev = np.eye(n)

        for task in tasks:
            J = task['J']
            x_des = task['x_des']
            x_dot_des = task.get('x_dot_des', np.zeros_like(x_des))
            x_ddot_des = task.get('x_ddot_des', np.zeros_like(x_des))
            J_dot = task.get('J_dot', np.zeros_like(J))
            Kp = task.get('Kp', 3.0)
            Kd = task.get('Kd', 0.3)

            # Task error in position and velocity
            # print("J_size", J)
            # print("q_curr",q_curr)
            x_curr = J @ q_curr
            x_dot_curr = J @ q_dot_curr
            e = x_des - x_curr
            e_dot = x_dot_des - x_dot_curr

            J_pre = J @ N_prev

            # Pseudo-inverse (can be dynamically consistent)
            
            J_pre_pinv_dc = self.pseudo_inverse(J_pre, method='dynamically_consistent', A=A)
            
            J_pre_pinv = self.pseudo_inverse(J_pre, method='svd')

            # Desired task-space acceleration (with feedback)
            x_ddot_cmd = x_ddot_des + Kp * e + Kd * e_dot

            # Projected Jacobian


            # Incremental joint command
            delta_q = delta_q_prev + J_pre_pinv @ (e - J @ delta_q_prev)        
            q_dot_cmd = q_dot_prev + J_pre_pinv @ (x_dot_des - J @ q_dot_prev)
            q_ddot_cmd = q_ddot_prev + J_pre_pinv_dc @ (x_ddot_cmd - J @ q_ddot_prev - J_dot @ q_dot_curr)
            # Update null-space projector for next task
            N_prev = N_prev @ (np.eye(n) - J_pre_pinv @ J_pre)
            delta_q_prev = delta_q
            q_dot_prev = q_dot_cmd
            q_ddot_prev = q_ddot_cmd

        # Compute final desired joint positions
        q_cmd = q_curr + delta_q

        return q_cmd, q_dot_cmd, q_ddot_cmd
