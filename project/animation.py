# animation.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.transforms as mtransforms

class VehicleAnimation:
    def __init__(self, ax, car_length=4.5, car_width=2.0, track=None):
        self.ax = ax
        self.car_length = car_length
        self.car_width = car_width
        self.track = track

        # Plot reference track
        if self.track is not None:
            self.ax.plot(self.track[:, 0], self.track[:, 1], '--k', alpha=0.7, label="Reference Track")

        # --- Vehicle Patches (no change) ---
        self.car_body = patches.Rectangle(
            (0, -self.car_width / 2.0),
            self.car_length, self.car_width,
            fc='blue', ec='black', alpha=0.8, label="Vehicle"
        )
        self.rear_axle_marker = patches.Circle((0, 0), radius=0.15, fc='red', label="Rear Axle (State Ref)")
        self.transform = mtransforms.Affine2D()
        self.car_body.set_transform(self.transform + self.ax.transData)
        self.rear_axle_marker.set_transform(self.transform + self.ax.transData)
        self.ax.add_patch(self.car_body)
        self.ax.add_patch(self.rear_axle_marker)
        # --- End Vehicle Patches ---

        # *** Add plot element for the predicted trajectory ***
        self.predicted_line, = self.ax.plot([], [], 'g.--', lw=1.5, alpha=0.8, label="Predicted Horizon")

        self.setup_legend() # Call legend setup

    def update(self, state):
        x, y, theta, _ = state
        self.transform.clear().rotate_deg(np.degrees(theta)).translate(x, y)

    # *** New method to update the predicted path plot ***
    def update_prediction(self, predicted_xy):
        """Updates the predicted trajectory line plot.

        Args:
            predicted_xy (np.ndarray): Array of predicted [x, y] coords (shape 2x(N+1))
                                       or None if prediction failed.
        """
        if predicted_xy is not None and predicted_xy.shape[0] == 2:
            self.predicted_line.set_data(predicted_xy[0, :], predicted_xy[1, :])
        else:
            # Clear the prediction if it failed or is invalid
            self.predicted_line.set_data([], [])

    def setup_plot(self, xlim, ylim):
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_xlabel("X coordinate (m)")
        self.ax.set_ylabel("Y coordinate (m)")
        # self.setup_legend() # Legend setup moved to end of init

    # Helper to setup legend (called after all plot elements are added)
    def setup_legend(self):
         handles, labels = self.ax.get_legend_handles_labels()
         if handles:
              self.ax.legend(handles, labels, loc='best', fontsize='small')