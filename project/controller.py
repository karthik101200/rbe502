import casadi as ca
import numpy as np

def mpc_controller(state, ref_traj, N=15, dt=0.1, wheelbase=2.5):
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

    rhs = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        (v / wheelbase) * ca.tan(delta),
        a
    )
    f = ca.Function('f', [states, controls], [rhs])

    opti = ca.Opti()
    X = opti.variable(n_states, N + 1)
    U = opti.variable(n_controls, N)
    X0 = opti.parameter(n_states)
    X_ref = opti.parameter(2, N)  # trajectory points

    opti.subject_to(X[:, 0] == X0)

    for k in range(N):
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

    # Cost weights
    Q = np.diag([10.0, 10.0]) # Reduced from 100
    Q_theta = 1.0            # Reduced from 10
    Q_v = 5.0
    R = np.diag([0.1, 1.0])   # Reduced from [1, 10]
    obj = 0

    opts = {
        "print_time": False,
        "ipopt.print_level": 3, # Increase print level (0-12, 3 or 5 is usually informative)
        "ipopt.max_iter": 300    # Increase max iterations slightly (default is often 3000, but ensure it's not too low)
        # "ipopt.acceptable_tol": 1e-4 # Optionally relax tolerance slightly for testing
    }
    opti.solver("ipopt", opts)
    for k in range(N):
        pos_error = X[0:2, k] - X_ref[:, k]
        desired_heading = ca.atan2(X_ref[1, k] - X[1, k], X_ref[0, k] - X[0, k])
        heading_error = ca.atan2(ca.sin(desired_heading - X[2, k]), ca.cos(desired_heading - X[2, k]))
        speed_error = X[3, k] - 3.0

        obj += ca.mtimes([pos_error.T, Q, pos_error])
        obj += Q_theta * heading_error**2
        obj += Q_v * speed_error**2
        obj += ca.mtimes([U[:, k].T, R, U[:, k]])

    opti.minimize(obj)

    opti.subject_to(opti.bounded(-2.0, U[0, :], 2.0))   # acceleration
    opti.subject_to(opti.bounded(-0.5, U[1, :], 0.5))   # steering
    opti.subject_to(opti.bounded(0, X[3, :], 10))       # speed

    opts = {"print_time": False, "ipopt.print_level": 0}
    opti.solver("ipopt", opts)

    opti.set_value(X0, state)
    opti.set_value(X_ref, ref_traj)

    # Inside mpc_controller function in controller.py
    try:
        sol = opti.solve()
        return np.array([sol.value(U[0, 0]), sol.value(U[1, 0])])
    except Exception as e: # Catch specific exception if possible, or general Exception
        print(f"!!! MPC Solver failed: {e}") # Add this print statement
        return np.array([0.0, 0.0])
