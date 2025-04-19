### main.py (Updated for trajectory tracking MPC)

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller
from path_planner import get_start_and_goal, generate_reference_trajectory
from config import TIMESTEP as DT, WHEELBASE

# Setup
state, goal = get_start_and_goal()
ref_path = generate_reference_trajectory(state, goal, N=200, dt=DT)
i = 0

# Plot setup
fig, ax = plt.subplots()
vis = VehicleAnimation(ax)
vis.setup_plot(ref_path, xlim=(-5, 155), ylim=(-5, 105))

# Set up video rendering
frames = []

# Simulation loop
trajectory = [state.copy()]
for i in tqdm(range(500)):
    # Get reference trajectory slice
    N = 30
    if i + N <= len(ref_path):
        path_segment = ref_path[i:i+N]
    else:
        # If near the end, pad with the last known point
        remaining = len(ref_path) - i
        if remaining > 0:
            pad = np.tile(ref_path[-1], (N - remaining, 1))
            path_segment = np.vstack((ref_path[i:], pad))
        else:
            path_segment = np.tile(ref_path[-1], (N, 1))
    if len(path_segment) < N:
        path_segment = np.pad(path_segment, ((0, N - len(path_segment)), (0, 0)), mode='edge')

    # try:
    control, predicted_path = mpc_controller(state, path_segment.T, N=N, dt=DT, wheelbase=WHEELBASE)
    # except:
    #     control = np.array([0.0, 0.0])
    #     predicted_path = None

    state = kinematic_bicycle_model(state, control, DT, WHEELBASE)
    trajectory.append(state.copy())

    vis.update(state, predicted_path=predicted_path[:2, :].T)  # Pass the predicted path to the animation

    # Render frame
    fig.canvas.draw()
    frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame_image)

    if np.linalg.norm(state[:2] - goal) < 0.5:
        break

# Save video
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
video_filename = os.path.join(output_dir, "trajectory_simulation.mp4")
height, width, _ = frames[0].shape
out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
