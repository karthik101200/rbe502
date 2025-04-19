# animation.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class VehicleAnimation:
    def __init__(self, ax, car_length=4.5, car_width=2.0):
        self.ax = ax
        self.car_length = car_length
        self.car_width = car_width

        self.car_patch = patches.Rectangle(
            (0, 0), car_length, car_width,
            fc='blue', ec='black', alpha=0.8
        )
        self.ax.add_patch(self.car_patch)
        self.predicted_path_line, = self.ax.plot([], [], 'g--', label="Predicted Path")

    def update(self, state, predicted_path=None):
        x, y, theta, _ = state

        rear_to_center = self.car_length / 2.0
        cx = x - rear_to_center * np.cos(theta)
        cy = y - rear_to_center * np.sin(theta)

        # Update car position and rotation
        self.car_patch.set_xy((cx, cy))
        self.car_patch.angle = np.degrees(theta)

        # Update predicted path if provided
        if predicted_path is not None:
            self.predicted_path_line.set_data(predicted_path[:, 0], predicted_path[:, 1])

    def setup_plot(self, ref_path, xlim, ylim):
        self.ax.plot(ref_path[:, 0], ref_path[:, 1], label="Reference Path", linestyle="--")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.legend()
