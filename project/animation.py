# animation.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class VehicleAnimation:
    def __init__(self, ax, car_length=4.5, car_width=2.0):
        self.ax = ax
        self.car_length = car_length
        self.car_width = car_width

        # Create the car rectangle (centered at rear axle)
        self.car_patch = patches.Rectangle(
            (0, 0), car_length, car_width,
            fc='blue', ec='black', alpha=0.8
        )
        self.ax.add_patch(self.car_patch)

    def update(self, state):
        x, y, theta, _ = state  # Ignore velocity for drawing

        # Shift car so that it's centered at rear axle
        rear_to_center = self.car_length / 2.0
        cx = x - rear_to_center * np.cos(theta)
        cy = y - rear_to_center * np.sin(theta)

        # Update car position and rotation
        self.car_patch.set_xy((cx, cy))
        self.car_patch.angle = np.degrees(theta)

    def setup_plot(self, xlim, ylim):
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
