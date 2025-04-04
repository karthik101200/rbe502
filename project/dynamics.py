# dynamics.py

import numpy as np

def kinematic_bicycle_model(state, control, dt, wheelbase):
    x, y, theta, v = state
    a, delta = control
    delta = np.clip(delta, -np.radians(45), np.radians(45))
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / wheelbase) * np.tan(delta) * dt
    v_next = v + a * dt

    return np.array([x_next, y_next, theta_next, v_next])
