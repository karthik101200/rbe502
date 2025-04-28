import casadi as ca
import numpy as np
from scipy.spatial.distance import cdist # Needed again for finding closest point
from config import TIMEDIFF as DT, WHEELBASE
# --- (Weights Q, R - same as before) ---
Q = np.diag([1.0, 1.0])
Q_theta = 1.0
Q_v = 10.0
R = np.diag([0.1, 1.0])

# *** Add track_width parameter and centerline ref ***
def mpc_controller(state, ref_traj_segment, N=15, dt=0.1, wheelbase=2.5,
                   centerline_ref_full=None, # Pass the full centerline
                   track_width=5.0, obstacle_safety_margin=1.0, predicted_obs_centers=None):         # Pass the track width
    # --- (Symbolic variables, Dynamics model f - same as before) ---

    car_length = WHEELBASE * 1.8
    car_width = WHEELBASE * 0.9
    ego_points_local = [
        ca.DM([0, -car_width / 2]), ca.DM([0, car_width / 2]),
        ca.DM([car_length, -car_width / 2]), ca.DM([car_length, car_width / 2]),
        ca.DM([car_length / 2, 0]),
    ]
    x, y, theta, v = ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('theta'), ca.MX.sym('v')
    states = ca.vertcat(x, y, theta, v)
    n_states = states.size()[0]
    a, delta = ca.MX.sym('a'), ca.MX.sym('delta')
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
    X_ref_segment = opti.parameter(2, N) # Reference segment for cost calculation
    obs_pred_params = []
    if predicted_obs_centers is not None:
        for i in range(len(predicted_obs_centers)):
            obs_pred_params.append(opti.parameter(2, N+1))
    opti.subject_to(X[:, 0] == X0)
    for k in range(N):
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

    obj = 0
    half_track_width = track_width / 2.0

    # Pre-calculate nearest centerline points if centerline_ref_full is provided
    # This lookup needs to be efficient; using numpy here for concept,
    # may need pure CasADi for true gradient-based optimization if path varies a lot.
    # For simplicity, we find the closest point based on the initial state `X0`
    # and assume the relevant centerline segment is near that index.
    # A more robust method involves CasADi functions or lookup tables.
    closest_center_index = 0
    if centerline_ref_full is not None:
         # Find index on full centerline closest to initial state for this horizon
         distances = cdist(state[:2].reshape(1,-1), centerline_ref_full)
         closest_center_index = np.argmin(distances)


    for k in range(N):
        # --- Cost Calculation (Use X_ref_segment for cost) ---
        pos_error = X[0:2, k] - X_ref_segment[:, k] # Error relative to N-step target segment
        # Heading/Speed error calculations same as before...
        # ... using X_ref_segment for desired heading calc ...
        desired_heading = ca.atan2(X_ref_segment[1, k] - X[1, k], X_ref_segment[0, k] - X[0, k])
        heading_error = ca.atan2(ca.sin(desired_heading - X[2, k]), ca.cos(desired_heading - X[2, k]))
        speed_error = X[3, k] - 4.0

        obj += ca.mtimes([pos_error.T, Q, pos_error])
        obj += Q_theta * heading_error**2
        obj += Q_v * speed_error**2
        obj += ca.mtimes([U[:, k].T, R, U[:, k]])

        # --- *** Add Boundary Constraints *** ---
        if centerline_ref_full is not None:
            # Approximate constraint: Keep predicted point within track width
            # relative to the 'closest' point on the full centerline segment.
            # This is an approximation; a more rigorous approach involves projecting
            # X[0:2, k] onto the centerline path segment.

            # Find the centerline point and tangent corresponding to step k in horizon
            # (Approximation: use index relative to closest point found earlier)
            current_center_index = min(closest_center_index + k, len(centerline_ref_full) - 1)
            center_pt = centerline_ref_full[current_center_index, :]

            # Get tangent direction (approximate using finite difference on centerline)
            next_center_index = min(current_center_index + 1, len(centerline_ref_full) - 1)
            prev_center_index = max(current_center_index - 1, 0)
            tangent_vec = centerline_ref_full[next_center_index, :] - centerline_ref_full[prev_center_index, :]
            tangent_norm = ca.norm_2(tangent_vec)
            # Avoid division by zero
            tangent_vec = ca.if_else(tangent_norm > 1e-6, tangent_vec / tangent_norm, ca.DM([1, 0]))


            # Get normal vector (rotate tangent 90 deg)
            normal_vec = ca.vertcat(-tangent_vec[1], tangent_vec[0])

            # Calculate vector from centerline point to predicted state point
            diff_vec = X[0:2, k] - center_pt

            # Project difference vector onto the normal vector -> lateral deviation
            # lateral_deviation = ca.dot(diff_vec, normal_vec) # This needs to be done carefully in CasADi
            # Simplified placeholder using numpy for illustration - NEEDS CasADi conversion
            # Note: This calculation MUST be done using CasADi symbolic operations
            #       to allow the solver to compute gradients. The numpy approach here
            #       is just conceptual. A proper CasADi implementation would likely
            #       use helper functions or perhaps linearize the boundary locally.
            # Placeholder: We'll constrain the distance for now as a simpler alternative.
            # This is NOT the same as lateral deviation but prevents going too far.
            dist_sq = ca.sumsqr(X[0:2, k] - center_pt)
            opti.subject_to(dist_sq <= (half_track_width * 1.5)**2) # Looser constraint: stay near centerline


            # *** Proper Lateral Deviation Constraint (Conceptual - Requires CasADi Implementation) ***
            # lateral_deviation = normal_vec[0]*diff_vec[0] + normal_vec[1]*diff_vec[1]
            # opti.subject_to(opti.bounded(-half_track_width, lateral_deviation, half_track_width))
        if predicted_obs_centers is not None:
            # For each step k in the prediction horizon            x_k = X[0, k]
            x_k = X[0, k]
            y_k = X[1, k]
            theta_k = X[2, k]
            cos_theta = ca.cos(theta_k)
            sin_theta = ca.sin(theta_k)
            global_points = []
            for pt_local in ego_points_local:
                x_gloval = cos_theta * pt_local[0] - sin_theta * pt_local[1] + x_k
                y_gloval = sin_theta * pt_local[0] + cos_theta * pt_local[1] + y_k
                global_points.append(ca.vertcat(x_gloval, y_gloval))
            
            for i, obs_pred_param in enumerate(obs_pred_params):
                obs_center_k = obs_pred_param[:, k]
                for pt_global in global_points:
                    dist = ca.norm_2(pt_global - obs_center_k)
                    opti.subject_to(dist >= obstacle_safety_margin**2)
    opti.minimize(obj)

    # --- (Control/State Limits - same as before) ---
    opti.subject_to(opti.bounded(-2.0, U[0, :], 2.0))
    opti.subject_to(opti.bounded(-0.5, U[1, :], 0.5))
    opti.subject_to(opti.bounded(0, X[3, :], 10))

    # --- Solver options / Set parameters / Solve ---
    opts = {
        "print_time": False,
        "ipopt.print_level": 0, # Set to 3 or 5 for more debug info if solver fails
        "ipopt.max_iter": 300
        # "ipopt.acceptable_tol": 1e-4 # Optionally relax tolerance slightly for testing
    }
    opti.solver("ipopt", opts)
    opti.set_value(X0, state)
    opti.set_value(X_ref_segment, ref_traj_segment) # Use the segment for cost
    if predicted_obs_centers is not None:
        for i, obs_pred_param in enumerate(obs_pred_params):
            if predicted_obs_centers[i].shape == (2, N+1):
                opti.set_value(obs_pred_param, predicted_obs_centers[i])
            else:
                print(f"!!! predicted_obs_centers[{i}] has unexpected shape: {predicted_obs_centers[i].shape}")
    try:
        sol = opti.solve()
        optimal_control = np.array([sol.value(U[0, 0]), sol.value(U[1, 0])])
        predicted_states_xy = sol.value(X)[0:2, :]
        return optimal_control, predicted_states_xy
    except Exception as e:
        print(f"!!! MPC Solver failed: {e}")
        # Add debug prints if needed
        return np.array([0.0, 0.0]), None