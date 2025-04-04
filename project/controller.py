# mpc_controller.py

import casadi as ca
import numpy as np

def mpc_controller(state, goal, N=20, dt=0.1, wheelbase=2.5):
    """
    MPC controller using kinematic bicycle model (CasADi).
    
    Params:
        state: current [x, y, theta, v]
        goal: [x_goal, y_goal]
        N: prediction horizon
        dt: timestep
        wheelbase: vehicle wheelbase
    
    Returns:
        optimal_control: [acceleration, steering]
    """
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    theta = ca.MX.sym('theta')
    v = ca.MX.sym('v')
    states = ca.vertcat(x, y, theta, v)
    n_states = states.size()[0]

    a = ca.MX.sym('a')
    delta = ca.MX.sym('delta')
    controls = ca.vertcat(a, delta)
    n_controls = controls.size()[0]

    # Define dynamics model
    rhs = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        (v / wheelbase) * ca.tan(delta),
        a
    )
    f = ca.Function('f', [states, controls], [rhs])

    # Optimization variables
    opti = ca.Opti()
    X = opti.variable(n_states, N + 1)
    U = opti.variable(n_controls, N)
    X0 = opti.parameter(n_states)
    X_ref = opti.parameter(2)

    # Initial state
    opti.subject_to(X[:, 0] == X0)

    # Dynamics constraints
    for k in range(N):
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

    # Objective function
    Q_goal = np.diag([30, 30])   # weight on goal
    R = np.diag([1, 1])          # weight on control effort
    obj = 0
    for k in range(N):
        pos_error = X[0:2, k] - X_ref
        if k == N - 1:
            obj += ca.mtimes([pos_error.T, 1000 * Q_goal, pos_error])
        else:
            obj += ca.mtimes([pos_error.T, Q_goal, pos_error])
        obj += ca.mtimes([U[:, k].T, R, U[:, k]])
    opti.minimize(obj)

    # Control constraints
    opti.subject_to(opti.bounded(-1.5, U[0, :], 1.5))   # accel
    opti.subject_to(opti.bounded(-0.4, U[1, :], 0.4))   # steering

    opti.subject_to(opti.bounded(0, X[3, :], 10))           # velocity limits

    # Solver
    opts = {"print_time": False, "ipopt.print_level": 0}
    opti.solver("ipopt", opts)

    # Provide current state and goal
    opti.set_value(X0, state)
    opti.set_value(X_ref, goal[:2])

    try:
        sol = opti.solve()
        a = sol.value(U[0, 0])
        delta = sol.value(U[1, 0])
        return np.array([a, delta])
    except RuntimeError:
        return np.array([0.0, 0.0])  # fallback in case solver fails
