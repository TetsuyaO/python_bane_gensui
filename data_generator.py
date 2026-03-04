# data_generator.py

import numpy as np
import tensorflow as tf

def FDM(init_x, init_v, init_t, gamma, omega, dt, T):
    '''
    Finite Difference Method for damped oscillator
    init_x: Initial position
    init_v: Initial velocity
    init_t: Initial time
    gamma: Damping coefficient / (2.0 * mass) -> mass assumed to be 1.0
    omega: Frequency
    '''
    x, v, t = init_x, init_v, init_t
    g, w0 = gamma, omega
    num_iter = int(T/dt)

    alpha = np.arctan(-1*g/np.sqrt(w0**2 - g**2))
    a = np.sqrt(w0**2 * x**2 / (w0**2 - g**2))

    t_array, x_array, v_array = [], [], []
    x_analytical_array, diff_array = [], []

    for i in range(num_iter):
        fx = v
        fv = -1*w0**2 * x - 2*g * v
        x = x + dt * fx
        v = v + dt * fv
        t = t + dt
        x_a = a * np.exp(-1*g * t) * np.cos(np.sqrt(w0**2 - g**2) * t + alpha)
        diff = x_a - x

        t_array.append(t)
        x_array.append(x)
        x_analytical_array.append(x_a)
        v_array.append(v)
        diff_array.append(diff)

    return t_array, x_array, v_array, x_analytical_array, diff_array

def analytical_solution(g, w0, t):
    '''
    Analytical solution for damped oscillator
    g: Damping coefficient / (2.0 * mass) -> mass assumed to be 1.0
    w0: Frequency
    t: Time points (tf.linspace)
    '''
    assert g <= w0
    w = np.sqrt(w0**2-g**2)
    phi = np.arctan(-g/w)
    A = 1/(2*np.cos(phi))
    cos = tf.math.cos(phi+w*t)
    sin = tf.math.sin(phi+w*t)
    exp = tf.math.exp(-g*t)
    x = exp*2*A*cos
    return x

def generate_training_data(g=2.0, w0=20.0, n_points=500):
    '''
    Generate training data for both DDNN and PINN
    '''
    # Generate time points
    t = tf.linspace(0, 1, n_points)
    t = tf.reshape(t, [-1, 1])
    
    # Generate analytical solution
    x = analytical_solution(g, w0, t)
    x = tf.reshape(x, [-1, 1])
    
    # Create data points for DDNN (evenly spaced)
    ddnn_indices = [i for i in range(0, 300, 20)]
    t_data_ddnn = tf.gather(t, ddnn_indices)
    x_data_ddnn = tf.gather(x, ddnn_indices)
    
    # Create data points for PINN (random points)
    pinn_indices = [0, 35, 50, 110, 300]
    t_data_pinn = tf.gather(t, pinn_indices)
    x_data_pinn = tf.gather(x, pinn_indices)
    
    # Generate PINN collocation points
    t_pinn = tf.linspace(0, 1, 30)
    t_pinn = tf.reshape(t_pinn, [-1, 1])
    
    return {
        't': t,
        'x': x,
        't_data_ddnn': t_data_ddnn,
        'x_data_ddnn': x_data_ddnn,
        't_data_pinn': t_data_pinn,
        'x_data_pinn': x_data_pinn,
        't_pinn': t_pinn,
        'c': 2*g,
        'k': w0**2
    }