# animation.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.transforms as mtransforms

class VehicleAnimation:
    # Add left/right boundary args
    def __init__(self, ax, car_length=4.5, car_width=2.0, num_obstacle_cars=0, track=None,
                 left_boundary=None, right_boundary=None): # Removed static_obstacle_cars
        self.ax = ax
        self.car_length = car_length # Ego car dimensions
        self.car_width = car_width   # Ego car dimensions
        self.num_obstacle_cars = num_obstacle_cars
        self.track = track
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        # Plot reference track (centerline)
        if self.track is not None:
            self.ax.plot(self.track[:, 0], self.track[:, 1], '--k', alpha=0.7, label="Reference Track")

        # *** Plot track boundaries if provided ***
        if self.left_boundary is not None:
            self.ax.plot(self.left_boundary[:, 0], self.left_boundary[:, 1], '-m', alpha=0.5, lw=1, label="Track Boundary")
        
        if self.right_boundary is not None:
            # Only add label once
            label = None if self.left_boundary is not None else "Track Boundary"
            self.ax.plot(self.right_boundary[:, 0], self.right_boundary[:, 1], '-m', alpha=0.5, lw=1, label=label)
    
        # --- Vehicle Patches (no change) ---
        self.car_body = patches.Rectangle(
            (0, -self.car_width / 2.0),
            self.car_length, self.car_width,
            fc='blue', ec='black', alpha=0.8, label="Vehicle"
        )
        self.rear_axle_marker = patches.Circle((0, 0), radius=0.15, fc='red', label="Rear Axle (State Ref)")
        self.transform_ego = mtransforms.Affine2D()
        self.car_body.set_transform(self.transform_ego + self.ax.transData)
        self.rear_axle_marker.set_transform(self.transform_ego + self.ax.transData)
        self.ax.add_patch(self.car_body)
        self.ax.add_patch(self.rear_axle_marker)

        # --- Predicted Line (no change) ---
        self.predicted_line, = self.ax.plot([], [], 'g.--', lw=1.5, alpha=0.8, label="Predicted Horizon")

        self.obstacle_car_patches = []
        self.obstacle_car_transforms = []
        # Use slightly different dimensions/color for obstacles for clarity
        obs_car_length = car_length * 0.95
        obs_car_width = car_width * 0.95

        for i in range(self.num_obstacle_cars):
             patch = patches.Rectangle(
                 (0, -obs_car_width / 2.0), # Relative to center (assuming state is center)
                 obs_car_length, obs_car_width,
                 fc='orange', ec='black', alpha=0.7, label="Obstacle Car" if i == 0 else ""
             )
             transform = mtransforms.Affine2D()
             patch.set_transform(transform + self.ax.transData)
             self.ax.add_patch(patch)
             self.obstacle_car_patches.append(patch)
             self.obstacle_car_transforms.append(transform)



        self.setup_legend() # Call legend setup

    def update(self, state):
        x, y, theta, _ = state
        self.transform_ego.clear().rotate_deg(np.degrees(theta)).translate(x, y)

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
    
    def update_obstacles(self, obstacle_states):
        """Updates the positions and orientations of the obstacle car patches.

        Args:
            obstacle_states (list): A list of numpy arrays, where each array is the
                                     state [x, y, theta, v] of an obstacle car.
        """
        num_to_update = min(len(obstacle_states), self.num_obstacle_cars)
        for i in range(num_to_update):
             x_obs, y_obs, theta_obs, _ = obstacle_states[i]
             self.obstacle_car_transforms[i].clear() \
                 .rotate_deg(np.degrees(theta_obs)) \
                 .translate(x_obs, y_obs)
        # Hide unused patches if fewer obstacles are passed than initialized
        for i in range(num_to_update, self.num_obstacle_cars):
             self.obstacle_car_patches[i].set_visible(False)
             
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