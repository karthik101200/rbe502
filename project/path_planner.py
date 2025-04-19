### path_planner.py

def get_start_and_goal():
    start = [0.0, 0.0, 0.0, 0.0]    # [x, y, theta, v]
    goal = [150.0, 100.0]             # [x_goal, y_goal]
    return start, goal

def generate_reference_trajectory(start, goal, N, dt, v_desired=3.0):
    import numpy as np
    start_pos = np.array(start[:2])
    goal_pos = np.array(goal)
    total_dist = np.linalg.norm(goal_pos - start_pos)
    total_time = total_dist / v_desired

    num_points = max(int(total_time / dt), N)
    xs = np.linspace(start_pos[0], goal_pos[0], num_points)
    ys = np.linspace(start_pos[1], goal_pos[1], num_points)

    return np.vstack((xs, ys)).T
