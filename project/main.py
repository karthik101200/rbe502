# main.py

import matplotlib.pyplot as plt
import numpy as np

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller
from path_planner import get_start_and_goal
from config import DT, WHEELBASE

# Setup
state, goal = get_start_and_goal()
trajectory = [state.copy()]

# Plot setup
fig, ax = plt.subplots()
vis = VehicleAnimation(ax)
vis.setup_plot(xlim=(-5, 100), ylim=(-5, 5))

# Simulation loop
for t in range(100):  # ~10 seconds
    control = mpc_controller(state, goal, dt=DT, wheelbase=WHEELBASE)
    state = kinematic_bicycle_model(state, control, DT, WHEELBASE)
    trajectory.append(state.copy())

    vis.update(state)
    plt.pause(0.05)

plt.show()
