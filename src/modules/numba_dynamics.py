"""
Numba-optimized neural network dynamics for high-performance simulation.
This module provides JIT-compiled functions for large-scale neural networks (10,000+ neurons).
"""

import numpy as np
from numba import jit, prange, types
from numba.typed import Dict
import numba

# Numba-optimized activation functions
@jit('float64(float64, float64, float64, float64)', nopython=True, cache=True)
def sigmoid_numba(x, r_m=1.0, beta=1.0, x_r=0.0):
    """Numba-optimized sigmoid function"""
    val = -beta * (x - x_r)
    # Manual clipping for Numba compatibility
    if val > 500.0:
        val = 500.0
    elif val < -500.0:
        val = -500.0
    return r_m / (1.0 + np.exp(val))

@jit('float64(float64, float64)', nopython=True, cache=True)
def relu_numba(x, amplitude=1.0):
    """Numba-optimized ReLU function"""
    if x > 0.0:
        return amplitude * x
    else:
        return 0.0

@jit('float64[:](float64[:], float64, float64)', nopython=True, cache=True)
def step_function_numba(x, q_f=1.0, x_f=0.0):
    """Numba-optimized step function"""
    result = np.zeros_like(x)
    for i in range(x.size):
        if x.flat[i] > x_f:
            result.flat[i] = q_f
        else:
            result.flat[i] = -(1.0 - q_f)
    return result

# Numba-optimized matrix operations
@jit('float64[:](float64[:,:], float64[:])', nopython=True, cache=True, parallel=True)
def matrix_vector_multiply_parallel(W, v):
    """Parallel matrix-vector multiplication optimized for large matrices"""
    N = W.shape[0]
    result = np.zeros(N)
    for i in prange(N):
        for j in range(N):
            result[i] += W[i, j] * v[j]
    return result

@jit('float64[:](float64[:], float64, float64, float64)', nopython=True, cache=True)
def apply_activation_sigmoid(u, r_m, beta, x_r):
    """Apply sigmoid activation to entire array"""
    result = np.zeros_like(u)
    for i in range(u.size):
        val = -beta * (u.flat[i] - x_r)
        # Manual clipping for Numba compatibility
        if val > 500.0:
            val = 500.0
        elif val < -500.0:
            val = -500.0
        result.flat[i] = r_m / (1.0 + np.exp(val))
    return result

@jit('float64[:](float64[:], float64)', nopython=True, cache=True)
def apply_activation_relu(u, amplitude):
    """Apply ReLU activation to entire array"""
    result = np.zeros_like(u)
    for i in range(u.size):
        if u.flat[i] > 0.0:
            result.flat[i] = amplitude * u.flat[i]
        else:
            result.flat[i] = 0.0
    return result

# Numba-optimized dynamics functions
@jit('float64[:](float64[:], float64[:,:], float64[:,:], float64, float64, int64, float64, float64, float64, float64)', nopython=True, cache=True)
def network_dynamics_recanatesi_numba(u, W_S, W_A, tau, zeta_value, 
                                      activation_type, r_m, beta, x_r, amplitude):
    """
    Numba-optimized Recanatesi dynamics:
    tau * du_i/dt = -u_i + sum_j W_S_ij * phi(u_j) + zeta(t) * sum_j W_A_ij * phi(u_j)
    """
    N = len(u)
    
    # Apply activation function
    if activation_type == 0:  # sigmoid
        phi_u = apply_activation_sigmoid(u, r_m, beta, x_r)
    else:  # relu
        phi_u = apply_activation_relu(u, amplitude)
    
    # Matrix-vector multiplications
    symmetric_input = matrix_vector_multiply_parallel(W_S, phi_u)
    asymmetric_input = matrix_vector_multiply_parallel(W_A, phi_u)
    
    # Compute derivative
    du_dt = np.zeros(N)
    for i in range(N):
        du_dt[i] = (-u[i] + symmetric_input[i] + zeta_value * asymmetric_input[i]) / tau
    
    return du_dt

@jit('float64[:](float64[:], float64[:,:], float64[:,:], float64, float64, int64, float64, float64, float64, float64)', nopython=True, cache=True)
def network_dynamics_brunel_numba(u, W_S, W_A, tau, zeta_value,
                                  activation_type, r_m, beta, x_r, amplitude):
    """
    Numba-optimized Brunel dynamics:
    tau * du_i/dt = -u_i + phi(sum_{j≠i} W^S_ij * u_j + zeta(t) * sum_{j≠i} W^A_ij * u_j)
    """
    N = len(u)
    
    # Matrix-vector multiplications
    symmetric_input = matrix_vector_multiply_parallel(W_S, u)
    asymmetric_input = matrix_vector_multiply_parallel(W_A, u)
    
    # Apply activation to total input
    total_input = symmetric_input + zeta_value * asymmetric_input
    
    if activation_type == 0:  # sigmoid
        activated_input = apply_activation_sigmoid(total_input, r_m, beta, x_r)
    else:  # relu
        activated_input = apply_activation_relu(total_input, amplitude)
    
    # Compute derivative
    du_dt = np.zeros(N)
    for i in range(N):
        du_dt[i] = (-u[i] + activated_input[i]) / tau
    
    return du_dt

@jit('float64[:](float64[:], float64, float64[:,:], float64[:,:], float64, float64, float64, int64, int64, float64, float64, float64, float64)', nopython=True, cache=True)
def rk4_step_numba(u, dt, W_S, W_A, tau, zeta_curr, zeta_next,
                   model_type, activation_type, r_m, beta, x_r, amplitude):
    """Numba-optimized RK4 integration step"""
    zeta_mid = 0.5 * (zeta_curr + zeta_next)
    
    # RK4 coefficients
    if model_type == 0:  # recanatesi
        k1 = dt * network_dynamics_recanatesi_numba(u, W_S, W_A, tau, zeta_curr,
                                                    activation_type, r_m, beta, x_r, amplitude)
        k2 = dt * network_dynamics_recanatesi_numba(u + 0.5*k1, W_S, W_A, tau, zeta_mid,
                                                    activation_type, r_m, beta, x_r, amplitude)
        k3 = dt * network_dynamics_recanatesi_numba(u + 0.5*k2, W_S, W_A, tau, zeta_mid,
                                                    activation_type, r_m, beta, x_r, amplitude)
        k4 = dt * network_dynamics_recanatesi_numba(u + k3, W_S, W_A, tau, zeta_next,
                                                    activation_type, r_m, beta, x_r, amplitude)
    else:  # brunel
        k1 = dt * network_dynamics_brunel_numba(u, W_S, W_A, tau, zeta_curr,
                                                activation_type, r_m, beta, x_r, amplitude)
        k2 = dt * network_dynamics_brunel_numba(u + 0.5*k1, W_S, W_A, tau, zeta_mid,
                                                activation_type, r_m, beta, x_r, amplitude)
        k3 = dt * network_dynamics_brunel_numba(u + 0.5*k2, W_S, W_A, tau, zeta_mid,
                                                activation_type, r_m, beta, x_r, amplitude)
        k4 = dt * network_dynamics_brunel_numba(u + k3, W_S, W_A, tau, zeta_next,
                                                activation_type, r_m, beta, x_r, amplitude)
    
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6.0

@jit('float64[:](int64, float64, float64, float64, float64)', nopython=True, cache=True)
def simulate_ou_process_numba(n_steps, dt, tau_zeta, zeta_bar, sigma_zeta):
    """Numba-optimized Ornstein-Uhlenbeck process simulation"""
    zeta = np.zeros(n_steps)
    zeta[0] = zeta_bar
    
    noise_scale = np.sqrt(2 * sigma_zeta**2 * tau_zeta * dt)
    
    for i in range(1, n_steps):
        dW = np.random.normal(0.0, 1.0)
        dzeta = (-zeta[i-1] + zeta_bar) * dt / tau_zeta + noise_scale * dW / tau_zeta
        zeta[i] = zeta[i-1] + dzeta
    
    return zeta

@jit('float64[:,:](float64[:,:], float64[:,:], float64[:], int64, float64, float64, int64, float64, float64, float64, float64, int64, float64[:])', nopython=True, cache=True)
def simulate_network_numba(W_S, W_A, initial_condition, n_steps, dt, tau,
                          activation_type, r_m, beta, x_r, amplitude,
                          model_type, zeta_array):
    """
    Main Numba-optimized network simulation function
    
    Parameters:
    -----------
    activation_type : int
        0 = sigmoid, 1 = relu
    model_type : int  
        0 = recanatesi, 1 = brunel
    """
    N = len(initial_condition)
    u = np.zeros((n_steps, N))
    u[0, :] = initial_condition
    
    for i in range(1, n_steps):
        zeta_curr = zeta_array[i-1]
        zeta_next = zeta_array[i] if i < len(zeta_array) else zeta_curr
        
        u[i, :] = rk4_step_numba(u[i-1, :], dt, W_S, W_A, tau, zeta_curr, zeta_next,
                                model_type, activation_type, r_m, beta, x_r, amplitude)
    
    return u

# Numba-optimized pattern overlap calculation
@jit('float64[:,:](float64[:,:], float64[:,:], float64[:], float64[:], int64, int64)', nopython=True, cache=True, parallel=True)
def calculate_pattern_overlaps_numba(u, patterns, phi_params, g_params,
                                     phi_type, g_type):
    """
    Numba-optimized pattern overlap calculation
    
    Parameters:
    -----------
    phi_type : int
        0 = sigmoid, 1 = relu
    g_type : int
        0 = sigmoid, 1 = step
    """
    n_timepoints, n_neurons = u.shape
    n_patterns = patterns.shape[0]
    overlaps = np.zeros((n_timepoints, n_patterns))
    
    # Pre-compute φ(η) and g(φ(η)) for all patterns
    phi_patterns = np.zeros_like(patterns)
    g_phi_patterns = np.zeros_like(patterns)
    
    for p in range(n_patterns):
        # Apply φ to pattern
        if phi_type == 0:  # sigmoid
            phi_patterns[p, :] = apply_activation_sigmoid(patterns[p, :], 
                                                         phi_params[0], phi_params[1], phi_params[2])
        else:  # relu
            phi_patterns[p, :] = apply_activation_relu(patterns[p, :], phi_params[3])
        
        # Apply g to φ(pattern)
        if g_type == 0:  # sigmoid
            g_phi_patterns[p, :] = apply_activation_sigmoid(phi_patterns[p, :],
                                                           g_params[0], g_params[1], g_params[2])
        else:  # step
            g_phi_patterns[p, :] = step_function_numba(phi_patterns[p, :], g_params[3], g_params[4])
    
    # Calculate overlaps for each timepoint
    for t in prange(n_timepoints):
        # Compute r = φ(state) for current timepoint
        if phi_type == 0:  # sigmoid
            r = apply_activation_sigmoid(u[t, :], phi_params[0], phi_params[1], phi_params[2])
        else:  # relu
            r = apply_activation_relu(u[t, :], phi_params[3])
        
        for p in range(n_patterns):
            g_phi_eta = g_phi_patterns[p, :]
            
            # Calculate variances
            var_g_phi_eta = np.var(g_phi_eta)
            var_r = np.var(r)
            
            if var_g_phi_eta > 1e-10 and var_r > 1e-10:
                # Calculate covariance manually
                mean_g = np.mean(g_phi_eta)
                mean_r = np.mean(r)
                cov = np.mean((g_phi_eta - mean_g) * (r - mean_r))
                
                # Calculate overlap
                overlaps[t, p] = cov / np.sqrt(var_g_phi_eta * var_r)
            else:
                overlaps[t, p] = 0.0
    
    return overlaps

# Performance monitoring utilities
def get_numba_performance_info():
    """Get information about Numba compilation and performance"""
    import numba
    return {
        'numba_version': numba.__version__,
        'threading_layer': numba.config.THREADING_LAYER,
        'parallel_support': numba.config.NUMBA_NUM_THREADS,
        'cuda_available': numba.cuda.is_available() if hasattr(numba, 'cuda') else False
    }

def estimate_memory_usage(N, n_steps):
    """Estimate memory usage for simulation"""
    # Main arrays: u (n_steps x N), W_S (N x N), W_A (N x N), zeta (n_steps)
    u_memory = n_steps * N * 8  # float64
    W_memory = 2 * N * N * 8   # W_S and W_A
    zeta_memory = n_steps * 8
    total_gb = (u_memory + W_memory + zeta_memory) / (1024**3)
    
    return {
        'total_gb': total_gb,
        'u_array_gb': u_memory / (1024**3),
        'connectivity_gb': W_memory / (1024**3),
        'recommended_max_neurons': int(np.sqrt(8 * 1024**3 / 8))  # 8GB limit for connectivity
    }