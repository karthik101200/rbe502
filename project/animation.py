# animation.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.transforms as mtransforms # <-- Correct import

class VehicleAnimation:
    def __init__(self, ax, car_length=4.5, car_width=2.0, track=None):
        self.ax = ax
        self.car_length = car_length
        self.car_width = car_width
        self.track = track # Store the reference track

        # Plot the track if provided
        if self.track is not None:
            self.ax.plot(self.track[:, 0], self.track[:, 1], '--k', alpha=0.7, label="Reference Track")

        # Initialize vehicle patch (center at rear axle for easier dynamics mapping)
        self.car_body = patches.Rectangle(
            (0, -self.car_width / 2.0), # Position relative to rear axle
            self.car_length, self.car_width,
            fc='blue', ec='black', alpha=0.8, label="Vehicle"
        )
        # Add a marker for the rear axle reference point
        self.rear_axle_marker = patches.Circle((0, 0), radius=0.15, fc='red', label="Rear Axle (State Ref)")

        # Use a transform for easier rotation/translation centered at the rear axle
        # Use the imported mtransforms alias here
        self.transform = mtransforms.Affine2D()
        self.car_body.set_transform(self.transform + self.ax.transData)
        self.rear_axle_marker.set_transform(self.transform + self.ax.transData)

        self.ax.add_patch(self.car_body)
        self.ax.add_patch(self.rear_axle_marker)


    def update(self, state):
        x, y, theta, _ = state # State represents the rear axle

        # Update the transform for the vehicle body and marker
        self.transform.clear() \
            .rotate_deg(np.degrees(theta)) \
            .translate(x, y)

    def setup_plot(self, xlim, ylim):
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_xlabel("X coordinate (m)")
        self.ax.set_ylabel("Y coordinate (m)")
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
             self.ax.legend(handles, labels, loc='best')