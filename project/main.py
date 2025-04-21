### main.py (Updated for Smoother Paths, Boundaries, Constraints)

import matplotlib.pyplot as plt
import numpy as np
import time
import os
import cv2
from tqdm import tqdm

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller # Imports the updated controller
from path_planner import generate_path # Imports the updated path generator
from config import TIMEDIFF as DT, WHEELBASE
from scipy.spatial.distance import cdist
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

# Create points for the controller (Example: just centers)
obstacle_points_for_controller = [{'x': obs['x'], 'y': obs['y'], 'radius': 0.1} # Treat center as tiny circle
                                   for obs in static_obstacle_cars]

obstacle_safety_margin = 3.0 # Meters

# *** Generate path and get boundaries ***
ref_path_centerline, start_state, estimated_duration, left_boundary, right_boundary = generate_path(
    path_type=selected_path_type,
    **path_params.get(selected_path_type, {})
)

state = start_state
current_time = 0.0
max_sim_steps = int(estimated_duration / DT) + 200
path_index = 0
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
                       obstacles = static_obstacle_cars,
                       car_length=WHEELBASE * 1.8, car_width=WHEELBASE * 0.9)
vis.setup_plot(xlim=xlim, ylim=ylim)
plt.title(f"MPC Bicycle Model Tracking: {selected_path_type.capitalize()} Path w/ Boundaries")
trajectory_line, = ax.plot([], [], 'b-', lw=1.5, label="Actual Trajectory")
trajectory_points = [state[:2].copy()]
vis.setup_legend()
frames = []

# --- Simulation Loop ---
start_loop_time = time.time()
lap_completed = False

for step in tqdm(range(max_sim_steps)):
    current_pos = state[:2]
    # Find closest point on CENTERLINE for target calculation
    distances = cdist(current_pos.reshape(1,-1), ref_path_centerline)
    closest_index = np.argmin(distances)
    path_index = closest_index # Use this as the reference progress index

    # Extract the reference path segment for the MPC COST FUNCTION
    ref_segment_cost = np.zeros((N, 2))
    for i in range(N):
        lookup_index = (path_index + i)
        # Loop or clamp index based on path type
        if selected_path_type in ["racetrack", "figure-eight"]:
             lookup_index %= len(ref_path_centerline)
        else:
            lookup_index = min(lookup_index, len(ref_path_centerline) - 1)
        ref_segment_cost[i, :] = ref_path_centerline[lookup_index]

    # *** Call MPC controller, passing the FULL centerline for boundary checks ***
    control, predicted_xy = mpc_controller(state,
                                           ref_segment_cost.T, # Segment for cost term
                                           N=N, dt=DT, wheelbase=WHEELBASE,
                                           centerline_ref_full=ref_path_centerline, # Full path for constraints
                                           track_width=track_width,
                                           obstacle_safety_margin=obstacle_safety_margin,
                                           obstacles=obstacle_points_for_controller) # Track width for constraints

    print(f"Control: {control}, Predicted Path: {predicted_xy}")

    # Update state using the bicycle model
    state = kinematic_bicycle_model(state, control, DT, WHEELBASE)

    # Store and update trajectory plot
    trajectory_points.append(state[:2].copy())
    if step > 0:
        trajectory_line.set_data(np.array(trajectory_points).T)

    # Update animations
    vis.update(state)
    vis.update_prediction(predicted_xy) # Update predicted path plot

    # Update plot title
    ax.set_title(f"MPC: {selected_path_type.capitalize()} (Time: {current_time:.2f}s)")

    # plt.pause(0.001)

    # Render frame
    fig.canvas.draw()
    frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame_image)
    current_time += DT

    # --- (Stopping Conditions - no change) ---
    # ...

end_loop_time = time.time()
# ... (print statements) ...

# plt.show()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
video_filename = os.path.join(output_dir, "trajectory_simulation.mp4")
height, width, _ = frames[0].shape
out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()