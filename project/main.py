### main.py (Updated for Smoother Paths, Boundaries, Constraints)

import matplotlib.pyplot as plt
import numpy as np
import time

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller # Imports the updated controller
from path_planner import generate_path # Imports the updated path generator
from config import TIMEDIFF as DT, WHEELBASE
from scipy.spatial.distance import cdist
from utils import get_rectangle_corners, obstacle_tracker


# --- Simulation Setup ---
selected_path_type = "racetrack" # Choose path type
track_width = 15.0                # Define track width

path_params = {
    # ... (params for straight, sinusoid - add track_width if needed) ...
    "sinusoid": {"length": 60, "amplitude": 6, "frequency": 0.15, "num_points": 600, "track_width": track_width},
    "racetrack": {"length": 60, "width": 30, "num_points": 500, "track_width": track_width},
    "figure-eight": {"scale": 20, "total_time_gen": 50, "track_width": track_width}
}

# In main.py
car_length_static = WHEELBASE * 1.8 # Or specific value
car_width_static = WHEELBASE * 0.9  # Or specific value
static_obstacle_cars = [
    {'x': 60, 'y': 0, 'theta': np.radians(10), 'length': car_length_static, 'width': car_width_static},
    {'x': -12, 'y': 20, 'theta': np.radians(10.5), 'length': car_length_static, 'width': car_width_static},
]

ref_path_centerline, my_start_state, estimated_duration, left_boundary, right_boundary = generate_path(
    path_type=selected_path_type,
    **path_params.get(selected_path_type, {})
)

my_state = my_start_state
my_target_speed = 4.0

### Obstacles
num_obstacles = 2
obstacle_states = []
obstacle_speed = [1.0, 1.0]
obstacle_threshold_distance = [5.0, 5.0]
obstacle_kp_vel = 1.0
obstacle_safety_margin = 1.8

start_index_obs1 = 50%len(ref_path_centerline)
start_index_obs2 = 100%len(ref_path_centerline)
obs1_start_state = np.array([ref_path_centerline[start_index_obs1][0], ref_path_centerline[start_index_obs1][1], 
                             np.arctan2(ref_path_centerline[start_index_obs1+1][1]-ref_path_centerline[start_index_obs1 -1][1], 
                             ref_path_centerline[start_index_obs1+1][0]-ref_path_centerline[start_index_obs1 - 1][0]),
                             obstacle_speed[0]])

obs2_start_state = np.array([ref_path_centerline[start_index_obs2][0], ref_path_centerline[start_index_obs2][1],
                             np.arctan2(ref_path_centerline[start_index_obs2+1][1]-ref_path_centerline[start_index_obs2-1][1], 
                             ref_path_centerline[start_index_obs2+1][0]-ref_path_centerline[start_index_obs2 - 1][0]),
                             obstacle_speed[1]])

obstacle_states = [obs1_start_state, obs2_start_state]

# Create points for the controller (Example: just centers)
# obstacle_points_for_controller = [{'x': obs['x'], 'y': obs['y'], 'radius': 0.1} # Treat center as tiny circle
#                                    for obs in static_obstacle_cars]

# # obstacle_safety_margin = 1.0 # Meters


# obstacle_corners = []

# for obs in static_obstacle_cars:
#     corners = get_rectangle_corners(obs['x'], obs['y'], obs['length'], obs['width'], obs['theta'])
#     obstacle_corners.append(corners)

# state = start_state
current_time = 0.0
max_sim_steps = int(estimated_duration / DT)*2
my_path_index = 0
N = 20

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(12, 9))
padding = track_width * 1.5 # Adjust padding based on track width
# Calculate limits based on centerline or boundaries if they exist
plot_path_for_limits = ref_path_centerline
if left_boundary is not None and right_boundary is not None:
    all_boundary_points = np.vstack((left_boundary, right_boundary))
    xlim = (np.min(all_boundary_points[:, 0]) - padding, np.max(all_boundary_points[:, 0]) + padding)
    ylim = (np.min(all_boundary_points[:, 1]) - padding, np.max(all_boundary_points[:, 1]) + padding)
else:
    xlim = (np.min(plot_path_for_limits[:, 0]) - padding, np.max(plot_path_for_limits[:, 0]) + padding)
    ylim = (np.min(plot_path_for_limits[:, 1]) - padding, np.max(plot_path_for_limits[:, 1]) + padding)


# *** Pass boundaries to animation ***
vis = VehicleAnimation(ax, track=ref_path_centerline,
                       left_boundary=left_boundary, right_boundary=right_boundary, # Pass boundaries
                       num_obstacle_cars = num_obstacles,
                       car_length=WHEELBASE * 1.8, car_width=WHEELBASE * 0.9)
vis.setup_plot(xlim=xlim, ylim=ylim)
plt.title(f"MPC Bicycle Model Tracking: {selected_path_type.capitalize()} Path w/ Boundaries")
trajectory_line, = ax.plot([], [], 'b-', lw=1.5, label="Actual Trajectory")
my_trajectory_points = [my_state[:2].copy()]
vis.setup_legend()

# --- Simulation Loop ---
start_loop_time = time.time()
lap_completed = False

for step in range(max_sim_steps):
    obstacle_controls = []
    for i in range(num_obstacles):
        control_observations = obstacle_tracker(
            obstacle_states[i], ref_path_centerline,
            obstacle_threshold_distance[i],
            obstacle_kp_vel, obstacle_speed[i],
            WHEELBASE
        )
        obstacle_controls.append(control_observations)
        obstacle_states[i] = kinematic_bicycle_model(
            obstacle_states[i], control_observations, DT, WHEELBASE
        )

    predicted_obs_centers = []
    temp_obs_centers = [s.copy() for s in obstacle_states]
    temp_obs_controls = [c.copy() for c in obstacle_controls]
    for i in range(num_obstacles):
        predicted_path_obs_i = np.zeros((2,N+1))
        predicted_path_obs_i[:,0] = temp_obs_centers[i][:2]

        current_pred_state = temp_obs_centers[i]
        for k in range(N):
            control_pred = obstacle_tracker(
                current_pred_state, ref_path_centerline,
                obstacle_threshold_distance[i],
                obstacle_kp_vel, obstacle_speed[i],
                WHEELBASE
            )

            current_pred_state = kinematic_bicycle_model(
                current_pred_state, control_pred, DT, WHEELBASE
            )

            predicted_path_obs_i[:,k+1] = current_pred_state[:2]

        predicted_obs_centers.append(predicted_path_obs_i)

    my_current_pos = my_state[:2]
    my_distances = cdist(my_current_pos.reshape(1,-1), ref_path_centerline)
    my_closest_index = np.argmin(my_distances)
    my_path_index = my_closest_index

    ref_segment_cost = np.zeros((N, 2))
    for i in range(N):
        lookup_index = (my_path_index + i) % len(ref_path_centerline)
        ref_segment_cost[i, :] = ref_path_centerline[lookup_index]
    

    # *** Call MPC controller, passing the FULL centerline for boundary checks ***
    my_control, my_predicted_xy = mpc_controller(my_state,
                                           ref_segment_cost.T, # Segment for cost term
                                           N=N, dt=DT, wheelbase=WHEELBASE,
                                           centerline_ref_full=ref_path_centerline, # Full path for constraints
                                           track_width=track_width,
                                           obstacle_safety_margin=obstacle_safety_margin,
                                           predicted_obs_centers=predicted_obs_centers) # Track width for constraints

    # Update state using the bicycle model
    my_state = kinematic_bicycle_model(my_state, my_control, DT, WHEELBASE)

    # Store and update trajectory plot
    my_trajectory_points.append(my_state[:2].copy())
    if step > 0:
        trajectory_line.set_data(np.array(my_trajectory_points).T)

    # Update animations
    vis.update(my_state)
    vis.update_prediction(my_predicted_xy) # Update predicted path plot
    vis.update_obstacles(obstacle_states) # Update obstacle positions

    # Update plot title
    ax.set_title(f"MPC: {selected_path_type.capitalize()} (Time: {current_time:.2f}s)")

    plt.pause(0.01)
    current_time += DT

    # --- (Stopping Conditions - no change) ---
    # ...

end_loop_time = time.time()
# ... (print statements) ...

plt.show()