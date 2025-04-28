import matplotlib.pyplot as plt
import numpy as np
import time
import math
from dynamics import kinematic_bicycle_model
from animation import VehicleAnimation
from controller import mpc_controller # Imports the updated controller
from path_planner import generate_path # Imports the updated path generator
from config import TIMEDIFF as DT, WHEELBASE
from scipy.spatial.distance import cdist

def get_rectangle_corners(x_center, y_center, length, width, angle_rad):
    """ Calculates the global coordinates of the four corners of a rotated rectangle. """
    # Corners relative to center (0,0) before rotation
    local_corners = [
        np.array([-length / 2, -width / 2]), # Bottom Left
        np.array([ length / 2, -width / 2]), # Bottom Right
        np.array([ length / 2,  width / 2]), # Top Right
        np.array([-length / 2,  width / 2])  # Top Left
    ]
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    global_corners = []
    for pt in local_corners:
        rotated_pt = rotation_matrix @ pt
        global_pt = rotated_pt + np.array([x_center, y_center])
        global_corners.append(global_pt.tolist()) # Convert to list for consistency if needed
    return global_corners # Returns list of [x,y] lists

def obstacle_tracker(vehicle_state, path_points, lookahead_dist, Kp_vel, target_vel, wheelbase):
    """
    Calculates control inputs (a, delta) for a vehicle to follow path_points.

    Args:
        vehicle_state: [x, y, theta, v]
        path_points: Nx2 array of path coordinates
        lookahead_dist: Distance to look ahead on the path
        Kp_vel: Proportional gain for velocity control
        target_vel: Desired velocity
        wheelbase: Vehicle wheelbase

    Returns:
        [acceleration, steering_angle]
    """
    x, y, theta, v = vehicle_state

    # Find the closest point on the path to the vehicle's current position
    distances = cdist(vehicle_state[:2].reshape(1,-1), path_points)
    closest_index = np.argmin(distances)

    # Find the lookahead point on the path
    lookahead_index = closest_index
    dist_to_lookahead = 0
    while dist_to_lookahead < lookahead_dist and lookahead_index < len(path_points) - 1:
        lookahead_index += 1
        dist_to_lookahead = np.linalg.norm(path_points[lookahead_index] - vehicle_state[:2])

    lookahead_point = path_points[lookahead_index]

    # Calculate target steering angle (delta) using Pure Pursuit
    alpha = math.atan2(lookahead_point[1] - y, lookahead_point[0] - x) - theta
    # Ensure lookahead distance is not zero if target is same as current pos
    L = max(np.linalg.norm(lookahead_point - vehicle_state[:2]), 0.1)
    delta = math.atan2(2.0 * wheelbase * math.sin(alpha), L)
    # Clamp steering angle (optional but good practice)
    delta = np.clip(delta, -np.radians(45), np.radians(45))


    # Calculate target acceleration (a) using simple P controller
    accel = Kp_vel * (target_vel - v)
    # Clamp acceleration (optional but good practice)
    accel = np.clip(accel, -2.0, 2.0) # Example limits

    return np.array([accel, delta])