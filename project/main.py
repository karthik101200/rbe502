### main.py (Updated for trajectory tracking MPC)

import matplotlib.pyplot as plt
import numpy as np

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller
from path_planner import get_start_and_goal, generate_reference_trajectory
from config import TIMEDIFF as DT, WHEELBASE

# Setup
state, goal = get_start_and_goal()
ref_path = generate_reference_trajectory(state, goal, N=100, dt=DT)
i = 0

# Plot setup
fig, ax = plt.subplots()
vis = VehicleAnimation(ax)
vis.setup_plot(xlim=(-5, 20), ylim=(-5, 20))

# Simulation loop
trajectory = [state.copy()]
for _ in range(200):
    # Get reference trajectory slice
    N = 15
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
    if len(path_segment) < 15:
        path_segment = np.pad(path_segment, ((0, 15 - len(path_segment)), (0, 0)), mode='edge')

    control = mpc_controller(state, path_segment.T, N=15, dt=DT, wheelbase=WHEELBASE)
    state = kinematic_bicycle_model(state, control, DT, WHEELBASE)
    trajectory.append(state.copy())

    vis.update(state)
    plt.pause(0.01)

    i += 1
    if np.linalg.norm(state[:2] - goal) < 0.5:
        break

plt.show()