# dynamics.py

import numpy as np

def kinematic_bicycle_model(state, control, dt, wheelbase):
    """
    Simulate one time step of the kinematic bicycle model.

    Parameters:
    - state: [x, y, theta, v] (position, heading, velocity)
    - control: [a, delta] (acceleration, steering angle)
    - dt: timestep (s)
    - wheelbase: distance between front and rear axle (m)

    Returns:
    - next_state: updated state after one time step
    """
    x, y, theta, v = state
    a, delta = control

    # Prevent tan(delta) from blowing up at ±90°
    delta = np.clip(delta, -np.radians(45), np.radians(45))

    # Kinematic bicycle model
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / wheelbase) * np.tan(delta) * dt
    v_next = v + a * dt

    return np.array([x_next, y_next, theta_next, v_next])
