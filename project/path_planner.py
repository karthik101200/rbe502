### path_planner.py (Updated with multiple path functions)
import numpy as np
from config import TIMEDIFF as DT

def get_default_start_state():
    """Returns a default starting state [x, y, theta, v]."""
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

def generate_straight_line_path(start_xy, goal_xy, num_points=200):
    """Generates a straight line path between two points."""
    xs = np.linspace(start_xy[0], goal_xy[0], num_points)
    ys = np.linspace(start_xy[1], goal_xy[1], num_points)
    path = np.vstack((xs, ys)).T
    print(f"Generated straight line path with {len(path)} points.")
    return path

def generate_sinusoid_path(length=50.0, amplitude=5.0, frequency=0.1, num_points=500):
    """Generates a sinusoidal path along the x-axis."""
    x = np.linspace(0, length, num_points)
    y = amplitude * np.sin(frequency * x)
    path = np.vstack((x, y)).T
    print(f"Generated sinusoid path with {len(path)} points.")
    return path

def generate_racetrack_path(length=40.0, width=20.0, curve_radius=10.0, num_points_straight=100, num_points_curve=100):
    """Generates a basic oval racetrack path (two straights, two semicircles)."""
    if curve_radius * 2 > width:
        raise ValueError("Curve radius is too large for the given width.")

    straight_len = length - 2 * curve_radius
    if straight_len < 0:
        raise ValueError("Length must be greater than 2 * curve_radius.")

    path_points = []

    # Start position (bottom right corner of the first straight)
    start_x = curve_radius
    start_y = 0

    # 1. First Straight (moving left)
    x_straight1 = np.linspace(start_x, start_x + straight_len, num_points_straight)
    y_straight1 = np.full(num_points_straight, start_y)
    path_points.append(np.vstack((x_straight1, y_straight1)).T)

    # 2. First Curve (top left semicircle)
    center_x1 = start_x + straight_len
    center_y1 = curve_radius
    angles1 = np.linspace(0, np.pi, num_points_curve) # 0 to 180 degrees
    x_curve1 = center_x1 + curve_radius * np.cos(angles1)
    y_curve1 = center_y1 + curve_radius * np.sin(angles1)
    path_points.append(np.vstack((x_curve1, y_curve1)).T)

    # 3. Second Straight (moving right)
    x_straight2 = np.linspace(center_x1 - curve_radius, start_x, num_points_straight)
    y_straight2 = np.full(num_points_straight, width)
    path_points.append(np.vstack((x_straight2, y_straight2)).T)

    # 4. Second Curve (bottom right semicircle)
    center_x2 = start_x
    center_y2 = curve_radius
    angles2 = np.linspace(np.pi, 2 * np.pi, num_points_curve) # 180 to 360 degrees
    x_curve2 = center_x2 + curve_radius * np.cos(angles2)
    y_curve2 = center_y2 + curve_radius * np.sin(angles2)
    path_points.append(np.vstack((x_curve2, y_curve2)).T)

    # Concatenate all parts
    path = np.concatenate(path_points, axis=0)

    # Ensure start and end points match approximately for closed loop
    # path = np.vstack((path, path[0])) # Optional: Explicitly close the loop

    print(f"Generated racetrack path with {len(path)} points.")
    return path

# --- Path Selection Function ---
def generate_path(path_type="racetrack", **kwargs):
    """Selects and generates the specified path type."""
    if path_type == "straight":
        start_xy = kwargs.get("start_xy", [0.0, 0.0])
        goal_xy = kwargs.get("goal_xy", [50.0, 5.0])
        num_points = kwargs.get("num_points", 200)
        start_state = get_default_start_state()
        start_state[0:2] = start_xy
        start_state[3] = 0.1
        # Calculate initial angle towards goal
        start_state[2] = np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
        path = generate_straight_line_path(start_xy, goal_xy, num_points)
        sim_duration_estimate = np.linalg.norm(np.array(goal_xy) - np.array(start_xy)) / 3.0 # Estimate time based on distance/speed

    elif path_type == "sinusoid":
        length = kwargs.get("length", 50.0)
        amplitude = kwargs.get("amplitude", 5.0)
        frequency = kwargs.get("frequency", 0.1)
        num_points = kwargs.get("num_points", 500)
        start_state = get_default_start_state() # Starts at (0,0) angle 0
        start_state[3] = 1
        path = generate_sinusoid_path(length, amplitude, frequency, num_points)
        sim_duration_estimate = length / 3.0 # Estimate time based on length/speed

    elif path_type == "racetrack":
        length = kwargs.get("length", 40.0)
        width = kwargs.get("width", 25.0)
        curve_radius = kwargs.get("curve_radius", 10.0)
        num_points_straight = kwargs.get("num_points_straight", 100)
        num_points_curve = kwargs.get("num_points_curve", 100)
        path = generate_racetrack_path(length, width, curve_radius, num_points_straight, num_points_curve)
        # Start state for racetrack: beginning of first straight, angle 0
        # In path_planner.py, inside generate_path function, when setting start_state
        # Example for racetrack:
        start_state = get_default_start_state()
        start_state[0] = path[0, 0]
        start_state[1] = path[0, 1]
        start_state[2] = 0.0
        start_state[3] = 0.1 # <-- Give small initial velocity (e.g., 0.1 m/s)
        # Estimate time based on approximate perimeter
        perimeter = 2 * (length - 2 * curve_radius) + 2 * np.pi * curve_radius
        sim_duration_estimate = perimeter / 3.0 * 1.2 # Add buffer time

    elif path_type == "figure-eight": # Keep the previous one as an option
        scale = kwargs.get("scale", 15.0)
        total_time_gen = kwargs.get("total_time_gen", 40.0) # Time used for generation, not sim
        t = np.arange(0, total_time_gen, DT)
        x = scale * np.sin(t * 2 * np.pi / total_time_gen)
        y = scale * np.sin(t * 4 * np.pi / total_time_gen) / 2
        path = np.vstack((x, y)).T
        start_state = np.array([path[0,0], path[0,1], np.pi/4, 0.1]) # Start at path beginning, angle pi/4
        
        print(f"Generated figure-eight path with {len(path)} points.")
        sim_duration_estimate = total_time_gen * 1.1 # Use generation time as estimate

    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    return path, start_state, sim_duration_estimate

# Example usage (optional, for testing)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test each path type
    path_types = ["straight", "sinusoid", "racetrack", "figure-eight"]
    params = {
        "straight": {"start_xy": [0,0], "goal_xy": [30, -10]},
        "sinusoid": {"length": 60, "amplitude": 8},
        "racetrack": {"length": 50, "width": 30, "curve_radius": 12},
        "figure-eight": {"scale": 20}
    }

    plt.figure(figsize=(12, 10))
    for i, p_type in enumerate(path_types):
        ax = plt.subplot(2, 2, i + 1)
        path, start, _ = generate_path(path_type=p_type, **params[p_type])
        ax.plot(path[:, 0], path[:, 1], '-r', label=f"Ref Path ({p_type})")
        ax.scatter(start[0], start[1], marker='o', s=100, color='g', label="Start", zorder=5)
        # Plot start direction arrow
        arrow_len = 2.0
        ax.arrow(start[0], start[1], arrow_len * np.cos(start[2]), arrow_len * np.sin(start[2]),
                 head_width=1.0, head_length=1.0, fc='g', ec='g', zorder=5)
        ax.set_title(f"{p_type.capitalize()} Path")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis('equal')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()