# main.py

import matplotlib.pyplot as plt
import numpy as np

from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller
from path_planner import get_start_and_goal
from config import TIMEDIFF, WHEELBASE

state, goal = get_start_and_goal()
trajectory = [state.copy()]

fig, ax = plt.subplots()
vis = VehicleAnimation(ax)
vis.setup_plot(xlim=(-5, 100), ylim=(-5, 5))
for t in range(100):  # ~10 seconds
    control = mpc_controller(state, goal, dt=TIMEDIFF, wheelbase=WHEELBASE)
    state = kinematic_bicycle_model(state, control, TIMEDIFF, WHEELBASE)
    trajectory.append(state.copy())

    vis.update(state)
    plt.pause(0.05)

plt.show()
