### path_planner.py (Updated with Smoother Racetrack & Boundary Calculation)
import numpy as np
from config import TIMEDIFF as DT
# *** Add scipy import for spline generation ***
from scipy import interpolate
from scipy.spatial.distance import cdist # For finding closest points efficiently

def get_default_start_state():
    """Returns a default starting state [x, y, theta, v]."""
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# --- (generate_straight_line_path and generate_sinusoid_path remain the same) ---
def generate_straight_line_path(start_xy, goal_xy, num_points=200):
    """Generates a straight line path between two points."""
    xs = np.linspace(start_xy[0], goal_xy[0], num_points)
    ys = np.linspace(start_xy[1], goal_xy[1], num_points)
    path = np.vstack((xs, ys)).T
    print(f"Generated straight line path with {len(path)} points.")
    return path, None, None # Return None for boundaries

def generate_sinusoid_path(length=50.0, amplitude=5.0, frequency=0.1, num_points=500, track_width=4.0):
    """Generates a sinusoidal path along the x-axis and its boundaries."""
    x = np.linspace(0, length, num_points)
    y = amplitude * np.sin(frequency * x)
    centerline = np.vstack((x, y)).T

    # Calculate path normals for boundaries
    dx = np.gradient(x)
    dy = np.gradient(y)
    normals = np.vstack((-dy, dx)).T
    norm_magnitudes = np.linalg.norm(normals, axis=1)
    # Avoid division by zero for straight sections
    valid_norms = norm_magnitudes > 1e-6
    unit_normals = np.zeros_like(normals)
    unit_normals[valid_norms] = normals[valid_norms] / norm_magnitudes[valid_norms, np.newaxis]

    half_width = track_width / 2.0
    left_boundary = centerline - unit_normals * half_width
    right_boundary = centerline + unit_normals * half_width

    print(f"Generated sinusoid path with {len(centerline)} points.")
    return centerline, left_boundary, right_boundary

# --- (Original generate_racetrack_path removed or renamed) ---

def generate_racetrack_spline(length=50.0, width=30.0, num_points=500, track_width=5.0, k=3, s=0):
    """
    Generates a smooth oval racetrack using B-splines and calculates boundaries.

    Args:
        length (float): Approximate length of the straights.
        width (float): Approximate width between straights.
        num_points (int): Number of points to generate for the final path.
        track_width (float): Total width of the track for boundaries.
        k (int): Degree of the spline. Cubic (k=3) is common.
        s (float): Smoothing factor for splprep. 0 means interpolate through points.

    Returns:
        tuple: (centerline, left_boundary, right_boundary) paths as Nx2 numpy arrays.
    """
    # Define key anchor points for an oval shape
    # Adjust points to influence shape (e.g., make straights longer/shorter)
    anchor_points = np.array([
        [0, 0],          # Start of bottom straight
        [length, 0],     # End of bottom straight / Start of right curve
        [length + width/2, width/2], # Mid point of right curve
        [length, width], # End of right curve / Start of top straight
        [0, width],      # End of top straight / Start of left curve
        [-width/2, width/2], # Mid point of left curve
        [0, 0]           # Close the loop
    ])

    # Use splprep to find the B-spline representation (parametric)
    # tck = tuple (t,c,k) containing the vector of knots, B-spline coefficients, and degree k
    # u = array of the parameters for each given point
    tck, u = interpolate.splprep([anchor_points[:, 0], anchor_points[:, 1]], s=s, per=True, k=k)

    # Evaluate the spline at a finer resolution (num_points)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = interpolate.splev(u_fine, tck)
    centerline = np.vstack((x_fine, y_fine)).T

    # Calculate path normals for boundaries
    # Use splev derivative capability: der=1
    dx_fine, dy_fine = interpolate.splev(u_fine, tck, der=1)
    normals = np.vstack((-dy_fine, dx_fine)).T
    norm_magnitudes = np.linalg.norm(normals, axis=1)
    valid_norms = norm_magnitudes > 1e-6 # Avoid division by zero
    unit_normals = np.zeros_like(normals)
    unit_normals[valid_norms] = normals[valid_norms] / norm_magnitudes[valid_norms, np.newaxis]

    half_width = track_width / 2.0
    left_boundary = centerline - unit_normals * half_width
    right_boundary = centerline + unit_normals * half_width

    print(f"Generated spline racetrack path with {len(centerline)} points.")
    return centerline, left_boundary, right_boundary, tck


# --- Path Selection Function (Updated) ---
def generate_path(path_type="racetrack", **kwargs):
    """Selects and generates the specified path type, including boundaries."""
    left_boundary, right_boundary = None, None # Default to no boundaries
    track_width = kwargs.get("track_width", 5.0) # Default track width if needed

    if path_type == "straight":
        # Boundaries not typically needed/defined for simple straight line goal
        start_xy = kwargs.get("start_xy", [0.0, 0.0])
        goal_xy = kwargs.get("goal_xy", [50.0, 5.0])
        num_points = kwargs.get("num_points", 200)
        path, _, _ = generate_straight_line_path(start_xy, goal_xy, num_points)
        start_state = get_default_start_state()
        start_state[0:2] = start_xy
        start_state[2] = np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
        sim_duration_estimate = np.linalg.norm(np.array(goal_xy) - np.array(start_xy)) / 3.0

    elif path_type == "sinusoid":
        length = kwargs.get("length", 50.0)
        amplitude = kwargs.get("amplitude", 5.0)
        frequency = kwargs.get("frequency", 0.1)
        num_points = kwargs.get("num_points", 500)
        path, left_boundary, right_boundary = generate_sinusoid_path(length, amplitude, frequency, num_points, track_width)
        start_state = get_default_start_state() # Starts at (0,0) angle 0
        sim_duration_estimate = length / 3.0

    elif path_type == "racetrack":
        length = kwargs.get("length", 50.0)
        width = kwargs.get("width", 30.0)
        num_points = kwargs.get("num_points", 500)
        path, left_boundary, right_boundary,tck = generate_racetrack_spline(length, width, num_points, track_width)
        # Start state for racetrack: beginning point from spline, angle based on initial direction
        start_state = get_default_start_state()
        start_state[0:2] = path[0, :]
        dx, dy = interpolate.splev(0, tck, der=1) # Get direction at start
        start_state[2] = np.arctan2(dy, dx)
        start_state[3] = 1.0
        # Estimate time based on approximate perimeter
        perimeter = 2 * length + np.pi * width # Rough estimate
        sim_duration_estimate = perimeter / 3.0 * 1.2

    elif path_type == "figure-eight": # Keep as option, add boundaries
        scale = kwargs.get("scale", 15.0)
        total_time_gen = kwargs.get("total_time_gen", 40.0)
        num_points = int(total_time_gen / DT)
        t = np.linspace(0, total_time_gen, num_points) # Use linspace for smoother time steps

        # Parametric equations
        x = scale * np.sin(t * 2 * np.pi / total_time_gen)
        y = scale * np.sin(t * 4 * np.pi / total_time_gen) / 2
        centerline = np.vstack((x, y)).T

        # Calculate derivatives and normals
        dt_val = t[1] - t[0]
        dx = np.gradient(x, dt_val)
        dy = np.gradient(y, dt_val)
        normals = np.vstack((-dy, dx)).T
        norm_magnitudes = np.linalg.norm(normals, axis=1)
        valid_norms = norm_magnitudes > 1e-6
        unit_normals = np.zeros_like(normals)
        unit_normals[valid_norms] = normals[valid_norms] / norm_magnitudes[valid_norms, np.newaxis]

        half_width = track_width / 2.0
        left_boundary = centerline - unit_normals * half_width
        right_boundary = centerline + unit_normals * half_width

        path = centerline # Rename for consistency
        start_state = np.array([path[0,0], path[0,1], np.arctan2(dy[0],dx[0]), 0.0]) # Use initial tangent
        print(f"Generated figure-eight path with {len(path)} points.")
        sim_duration_estimate = total_time_gen * 1.1

    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    return path, start_state, sim_duration_estimate, left_boundary, right_boundary

# --- (Example usage block can be updated to plot boundaries too) ---