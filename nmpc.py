import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

def create_ipm_nmpc(N, dt, n_footholds, K1=1.0, K2=1.0, K3=0.2, K4=100.0, K5=1.0):
    """
    Create an ACADOS NMPC solver for the Inverted Pendulum Model (IPM)
    
    Parameters:
    -----------
    N : int
        Prediction horizon steps
    dt : float
        Sampling time
    n_footholds : int
        Number of stance feet (4 for quadruped)
    K1-K5 : float
        Cost function weights (tune according to your robot)
        
    Returns:
    --------
    nmpc_solver : AcadosOcpSolver
        ACADOS NMPC solver object
    ocp : AcadosOcp
        OCP object (useful to update references)
    """
    
    # ------------------------
    # 1. Define state and input dimensions
    # ------------------------
    n_state = 6  # [r_x, r_y, r_z, v_x, v_y, v_z]
    n_control = 1 + n_footholds  # [¨h, w1, w2, ..., w_n_footholds]
    
    # Define CasADi symbols
    x = ca.SX.sym('x', n_state)
    u = ca.SX.sym('u', n_control)
    s = ca.SX.sym('s', 3*n_footholds)  # foot positions in world frame
    
    # ------------------------
    # 2. Define IPM dynamics
    # ------------------------
    r = x[:3]      # CoM position
    v = x[3:]      # CoM velocity
    h_ddot = u[0]  # vertical acceleration input
    w = u[1:]      # foot weights
    
    # Convex combination to get CoP
    p = ca.SX.zeros(3)
    for i in range(n_footholds):
        p += w[i] * s[3*i:3*i+3]  # s_i = [x, y, z]
    
    g = ca.SX([0, 0, -9.81])  # gravity
    r_ddot = (r - p) * h_ddot + g + ca.vertcat(0,0,0)  # simplified horizontal terms
    
    # Semi-implicit Euler
    x_next = ca.vertcat(r + v*dt + 0.5*r_ddot*dt**2, v + r_ddot*dt)
    
    # Define CasADi function for dynamics
    f_ipm = ca.Function('f_ipm', [x, u, s], [x_next])
    
    # ------------------------
    # 3. Set up ACADOS OCP
    # ------------------------
    ocp = AcadosOcp()
    ocp.model.name = 'IPM_model'
    
    # Dynamics
    from acados_template import AcadosModel
    model = AcadosModel()
    model.x = x
    model.u = u
    model.p = s
    model.f_expl_expr = x_next  # explicit dynamics
    model.name = 'ipm_model'
    ocp.model = model
    
    # Prediction horizon
    ocp.dims.N = N
    
    # ------------------------
    # 4. Cost function (user can update references)
    # ------------------------
    # Quadratic cost: track CoM position & velocity + soft weights
    # Create weight matrices
    Q = ca.diag([K1, K1, K2, 0.1, 0.1, 0.1])  # x,y,z,vx,vy,vz
    R = ca.diag([0.01] + [K3]*n_footholds)    # u: ¨h + w_i
    
    # References (to be updated online)
    x_ref = ca.SX.sym('x_ref', n_state)
    u_ref = ca.SX.sym('u_ref', n_control)
    
    ocp.cost.W_e = Q  # terminal cost
    ocp.cost.W = Q    # stage cost
    ocp.cost.W_u = R
    
    ocp.cost.yref_e = x_ref
    ocp.cost.yref = x_ref
    
    # ------------------------
    # 5. Constraints
    # ------------------------
    # Foot weights >= 0
    u_min = [ -ca.inf ] + [0.0]*n_footholds
    u_max = [ ca.inf ] + [1.0]*n_footholds
    ocp.constraints.lbu = u_min
    ocp.constraints.ubu = u_max
    ocp.constraints.idxbu = list(range(n_control))
    
    # Optional: sum(w_i) ≈ 1 enforced softly in cost
    
    # ------------------------
    # 6. Solver options
    # ------------------------
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # Sparse GN
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.print_level = 0
    
    # ------------------------
    # 7. Create ACADOS solver
    # ------------------------
    nmpc_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    
    return nmpc_solver, ocp, f_ipm


# Parameters
# N = 20            # horizon
# dt = 0.02         # 50 Hz
# n_footholds = 4   # quadruped

# # Create NMPC solver
# nmpc_solver, ocp, f_ipm = create_ipm_nmpc(N, dt, n_footholds)

# # Simulation loop
# for t in range(sim_steps):
#     # 1. Get current state from MuJoCo
#     x0 = mujoco.get_com_position_velocity()  # shape (6,)
    
#     # 2. Set references based on user command
#     r_ref = user_command_position()           # shape (3,)
#     v_ref = [0,0,0]
#     x_ref = r_ref + v_ref
    
#     # 3. Update solver references
#     nmpc_solver.set(0, "yref", x_ref)
    
#     # 4. Solve NMPC
#     status = nmpc_solver.solve()
#     u_opt = nmpc_solver.get(0, "u")
    
#     # 5. Apply control to MuJoCo (e.g., convert ¨h + w_i → forces/torques)
#     mujoco.apply_ipm_control(u_opt)
    
#     # 6. Step simulation
#     mujoco.step()
