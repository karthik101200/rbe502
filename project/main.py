### main.py (Updated for Multiple Paths & Auto Limits)

import matplotlib.pyplot as plt
import numpy as np
import time

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller
# Import the main path generation function
from path_planner import generate_path
from config import TIMEDIFF as DT, WHEELBASE

# --- Simulation Setup ---

# <<< --- Select Path Type --- >>>
# Options: "straight", "sinusoid", "racetrack", "figure-eight"
selected_path_type = "racetrack"

# <<< --- Path Specific Parameters (optional) --- >>>
path_params = {
    "straight": {"start_xy": [0, 0], "goal_xy": [40, 10], "num_points": 300},
    "sinusoid": {"length": 60, "amplitude": 6, "frequency": 0.15, "num_points": 600},
    "racetrack": {"length": 60, "width": 30, "curve_radius": 12, "num_points_straight": 150, "num_points_curve": 150},
    "figure-eight": {"scale": 20, "total_time_gen": 50}
}

# Generate the selected path and get start state + estimated duration
ref_path, start_state, estimated_duration = generate_path(
    path_type=selected_path_type,
    **path_params.get(selected_path_type, {}) # Pass specific params if defined
)

state = start_state # Use the specific start state for the path
current_time = 0.0
# Use estimated duration to set max steps, add buffer
max_sim_steps = int(estimated_duration / DT) + 100
path_index = 0 # Tracks progress along the reference path

# MPC settings
N = 15 # Prediction horizon

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(12, 9)) # Adjusted figure size

# Calculate plot limits dynamically based on the path
padding = 8.0 # Increase padding for better visibility
xlim = (np.min(ref_path[:, 0]) - padding, np.max(ref_path[:, 0]) + padding)
ylim = (np.min(ref_path[:, 1]) - padding, np.max(ref_path[:, 1]) + padding)

# Pass the generated track to the animation class
vis = VehicleAnimation(ax, track=ref_path,
                       car_length=WHEELBASE * 1.8, car_width=WHEELBASE * 0.9) # Adjusted car size

# Set up plot with dynamically calculated limits
vis.setup_plot(xlim=xlim, ylim=ylim)
plt.title(f"MPC Bicycle Model Tracking: {selected_path_type.capitalize()} Path")

# Plot the actual trajectory the vehicle follows
trajectory_line, = ax.plot([], [], 'b-', lw=1.5, label="Actual Trajectory") # Line object for trajectory
trajectory_points = [state[:2].copy()] # Start trajectory plot from initial state

# --- Simulation Loop ---
start_loop_time = time.time()
lap_completed = False # For closed tracks

for step in range(max_sim_steps):
    # Find the closest point on the reference path to the current state's rear axle
    current_pos = state[:2]
    print(f"Current Position: {current_pos}")
    distances = np.linalg.norm(ref_path - current_pos, axis=1)
    closest_index = np.argmin(distances)

    # Use the index of the closest point to define the start of the MPC reference segment
    # This helps keep the vehicle oriented towards the upcoming path section
    path_index = closest_index

    # Extract the reference path segment for the MPC horizon
    ref_segment = np.zeros((N, 2)) # Initialize segment array
    for i in range(N):
        # Use modulo arithmetic for closed tracks (like racetrack) to loop around
        lookup_index = (path_index + i)
        if selected_path_type in ["racetrack", "figure-eight"]: # Add other closed loops if needed
             lookup_index %= len(ref_path)
        else: # For open tracks, stop at the end
            lookup_index = min(lookup_index, len(ref_path) - 1)
        ref_segment[i, :] = ref_path[lookup_index]


    # Get control input from MPC
    # MPC expects reference as shape (2, N)
    control = mpc_controller(state, ref_segment.T, N=N, dt=DT, wheelbase=WHEELBASE)

    # Update state using the bicycle model
    prev_state = state.copy()
    state = kinematic_bicycle_model(state, control, DT, WHEELBASE)

    # Store and update trajectory plot
    trajectory_points.append(state[:2].copy())
    # Only plot every Nth point or so to avoid overly dense lines if needed
    # if step % 2 == 0:
    #     trajectory_line.set_data(np.array(trajectory_points).T)
    # More efficient plotting:
    if step > 0: # Avoid plotting single point lines
        trajectory_line.set_data(np.array(trajectory_points).T)

    # Update animation
    vis.update(state)

    # Update plot title with time
    ax.set_title(f"MPC Bicycle Model Tracking: {selected_path_type.capitalize()} (Time: {current_time:.2f}s)")

    plt.pause(0.001) # Reduce pause for potentially faster simulation visualization

    current_time += DT

    # --- Stopping Conditions ---
    # 1. Reached end of path index (for open paths)
    if selected_path_type in ["straight", "sinusoid"] and path_index >= len(ref_path) - 2:
         print(f"Reached end of {selected_path_type} path.")
         break

    # 2. Lap completion (for closed paths) - Simple check: pass start after sufficient steps
    if selected_path_type in ["racetrack", "figure-eight"]:
        dist_to_start = np.linalg.norm(state[:2] - ref_path[0])
        # Check if we've moved significantly away initially and then returned close to start
        if step > len(ref_path) // 4 and dist_to_start < 1.0 and not lap_completed:
             print(f"Completed one lap (approximately) on {selected_path_type}.")
             lap_completed = True # Set flag to allow breaking later or continue for more laps
             # break # Uncomment to stop after one lap

    # 3. Timeout (Fallback) - loop finishes naturally

end_loop_time = time.time()
sim_duration = end_loop_time - start_loop_time
print(f"\nSimulation loop finished in {sim_duration:.2f} seconds.")
print(f"Simulated {current_time:.2f} seconds of vehicle time ({step+1} steps).")

# Keep plot open
plt.show()