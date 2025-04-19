### main.py (Updated for Multiple Paths & Auto Limits & Prediction Viz)

import matplotlib.pyplot as plt
import numpy as np
import time

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller # Imports the updated controller
from path_planner import generate_path
from config import TIMEDIFF as DT, WHEELBASE

# --- Simulation Setup ---
selected_path_type = "straight" # Choose path type
path_params = {
    "straight": {"start_xy": [0, 0], "goal_xy": [40, 10], "num_points": 300},
    "sinusoid": {"length": 60, "amplitude": 6, "frequency": 0.15, "num_points": 600},
    "racetrack": {"length": 60, "width": 30, "curve_radius": 12, "num_points_straight": 150, "num_points_curve": 150},
    "figure-eight": {"scale": 20, "total_time_gen": 50}
}
ref_path, start_state, estimated_duration = generate_path(
    path_type=selected_path_type,
    **path_params.get(selected_path_type, {})
)
state = start_state
current_time = 0.0
max_sim_steps = int(estimated_duration / DT) + 200 # Add more buffer maybe
path_index = 0
N = 45 # Prediction horizon

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(12, 9))
padding = 8.0
xlim = (np.min(ref_path[:, 0]) - padding, np.max(ref_path[:, 0]) + padding)
ylim = (np.min(ref_path[:, 1]) - padding, np.max(ref_path[:, 1]) + padding)
vis = VehicleAnimation(ax, track=ref_path,
                       car_length=WHEELBASE * 1.8, car_width=WHEELBASE * 0.9)
vis.setup_plot(xlim=xlim, ylim=ylim) # Setup plot limits etc.
plt.title(f"MPC Bicycle Model Tracking: {selected_path_type.capitalize()} Path")
trajectory_line, = ax.plot([], [], 'b-', lw=1.5, label="Actual Trajectory")
trajectory_points = [state[:2].copy()]
vis.setup_legend() # Update legend now that actual trajectory line is added

# --- Simulation Loop ---
start_loop_time = time.time()
lap_completed = False

for step in range(max_sim_steps):
    current_pos = state[:2]
    distances = np.linalg.norm(ref_path - current_pos, axis=1)
    closest_index = np.argmin(distances)
    path_index = closest_index

    ref_segment = np.zeros((N, 2))
    for i in range(N):
        lookup_index = (path_index + i)
        if selected_path_type in ["racetrack", "figure-eight"]:
             lookup_index %= len(ref_path)
        else:
            lookup_index = min(lookup_index, len(ref_path) - 1)
        ref_segment[i, :] = ref_path[lookup_index]

    # *** Get control input AND predicted trajectory from MPC ***
    control, predicted_xy = mpc_controller(state, ref_segment.T, N=N, dt=DT, wheelbase=WHEELBASE)

    # Update state using the bicycle model
    prev_state = state.copy()
    state = kinematic_bicycle_model(state, control, DT, WHEELBASE)

    # Store and update trajectory plot
    trajectory_points.append(state[:2].copy())
    if step > 0:
        trajectory_line.set_data(np.array(trajectory_points).T)

    # Update base vehicle animation
    vis.update(state)
    # *** Update the prediction visualization ***
    vis.update_prediction(predicted_xy)

    # Update plot title
    ax.set_title(f"MPC Bicycle Model Tracking: {selected_path_type.capitalize()} (Time: {current_time:.2f}s)")

    plt.pause(0.001)
    current_time += DT

    # --- Stopping Conditions ---
    # (No change here)
    if selected_path_type in ["straight", "sinusoid"] and path_index >= len(ref_path) - 2:
         print(f"Reached end of {selected_path_type} path.")
         break
    if selected_path_type in ["racetrack", "figure-eight"]:
        dist_to_start = np.linalg.norm(state[:2] - ref_path[0])
        if step > len(ref_path) // 4 and dist_to_start < 1.0 and not lap_completed:
             print(f"Completed one lap (approximately) on {selected_path_type}.")
             lap_completed = True
             # break # Uncomment to stop after one lap

end_loop_time = time.time()
sim_duration = end_loop_time - start_loop_time
print(f"\nSimulation loop finished in {sim_duration:.2f} seconds.")
print(f"Simulated {current_time:.2f} seconds of vehicle time ({step+1} steps).")

plt.show()