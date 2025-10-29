"""
Numba-optimized neural network dynamics for high-performance simulation.
This module provides JIT-compiled functions for large-scale neural networks (>1000 neurons).
"""

import numpy as np
from numba import jit, prange

# -----------------------------
# Activation functions (Numba)
# -----------------------------

@jit(nopython=True, cache=False)
def sigmoid_numba(x, r_m=1.0, beta=1.0, x_r=0.0):
    val = -beta * (x - x_r)
    if val > 500.0:
        val = 500.0
    elif val < -500.0:
        val = -500.0
    return r_m / (1.0 + np.exp(val))

@jit(nopython=True, cache=False)
def relu_numba(x, amplitude=1.0):
    return amplitude * x if x > 0.0 else 0.0

@jit(nopython=True, cache=False)
def apply_activation_sigmoid(u, r_m, beta, x_r):
    N = u.size
    result = np.empty(N, dtype=np.float32)
    for i in range(N):
        val = -beta * (u[i] - x_r)
        if val > 500.0:
            val = 500.0
        elif val < -500.0:
            val = -500.0
        result[i] = r_m / (1.0 + np.exp(val))
    return result

@jit(nopython=True, cache=False)
def step_function_numba(x, q_f=1.0, x_f=0.0):
    N = x.size
    result = np.empty(N, dtype=np.float32)
    for i in range(N):
        result[i] = q_f if x[i] > x_f else -(1.0 - q_f)
    return result

# -----------------------------
# Utility: Parallel matrix-vector multiply
# -----------------------------

@jit(nopython=True, cache=True, parallel=True)
def matrix_vector_multiply_parallel(W, v):
    N = W.shape[0]
    result = np.zeros(N, dtype=np.float32)
    for i in prange(N):
        for j in range(N):
            result[i] += W[i, j] * v[j]
    return result

# -----------------------------
# Network dynamics
# -----------------------------

@jit(nopython=True, cache=False)
def network_dynamics_recanatesi_numba(u, W_S, W_A, tau, zeta_value, dt, r_m, beta, x_r):
    u = u.astype(np.float32)
    W_S = W_S.astype(np.float32)
    W_A = W_A.astype(np.float32)
    tau = np.float32(tau)
    dt = np.float32(dt)
    r_m = np.float32(r_m)
    beta = np.float32(beta)
    x_r = np.float32(x_r)
    
    N = u.size
    phi_u = apply_activation_sigmoid(u, r_m, beta, x_r)
    symmetric_input = matrix_vector_multiply_parallel(W_S, phi_u)
    asymmetric_input = matrix_vector_multiply_parallel(W_A, phi_u)
    du_dt = np.empty(N, dtype=np.float32)
    for i in range(N):
        du_dt[i] = (-u[i] + symmetric_input[i] + zeta_value * asymmetric_input[i]) / tau
    return du_dt

@jit(nopython=True, cache=False)
def network_dynamics_brunel_numba(u, W_S, W_A, tau, zeta_value, r_m, beta, x_r):
    u = u.astype(np.float32)
    W_S = W_S.astype(np.float32)
    W_A = W_A.astype(np.float32)
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)
    r_m = np.float32(r_m)
    beta = np.float32(beta)
    x_r = np.float32(x_r)
    
    symmetric_input = matrix_vector_multiply_parallel(W_S, u)
    asymmetric_input = matrix_vector_multiply_parallel(W_A, u)
    total_input = symmetric_input + zeta_value * asymmetric_input
    activated_input = apply_activation_sigmoid(total_input, r_m, beta, x_r)
    du_dt = (-u + activated_input) / tau
    return du_dt

# -----------------------------
# RK4 integration step (for network)
# -----------------------------

@jit(nopython=True, cache=False)
def rk4_step_numba(u, dt, W_S, W_A, tau, zeta_curr, zeta_next,
                   model_type, r_m, beta, x_r):

    zeta_mid = 0.5 * (zeta_curr + zeta_next)

    if model_type == 0:
        k1 = network_dynamics_recanatesi_numba(u, W_S, W_A, tau, zeta_curr, dt, r_m, beta, x_r)
        k2 = network_dynamics_recanatesi_numba(u + 0.5 * k1, W_S, W_A, tau, zeta_mid, dt, r_m, beta, x_r)
        k3 = network_dynamics_recanatesi_numba(u + 0.5 * k2, W_S, W_A, tau, zeta_mid, dt, r_m, beta, x_r)
        k4 = network_dynamics_recanatesi_numba(u + k3, W_S, W_A, tau, zeta_next, dt, r_m, beta, x_r)
    else:
        k1 = network_dynamics_brunel_numba(u, W_S, W_A, tau, zeta_curr, r_m, beta, x_r)
        k2 = network_dynamics_brunel_numba(u + 0.5 * k1, W_S, W_A, tau, zeta_mid, r_m, beta, x_r)
        k3 = network_dynamics_brunel_numba(u + 0.5 * k2, W_S, W_A, tau, zeta_mid, r_m, beta, x_r)
        k4 = network_dynamics_brunel_numba(u + k3, W_S, W_A, tau, zeta_next, r_m, beta, x_r)

    return u + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

# -----------------------------
# Ornstein-Uhlenbeck process (Heun)
# -----------------------------

@jit(nopython=True, cache=False)
def simulate_ou_process_numba(n_steps, dt, tau_zeta, zeta_bar, sigma_zeta):
    """
    Simulate an Ornstein-Uhlenbeck process with Heun's method (2nd order stochastic).
    """
    zeta = np.empty(n_steps, dtype=np.float32)
    zeta[0] = np.float32(zeta_bar)

    for i in range(1, n_steps):
        z_curr = zeta[i - 1]
        dW = np.float32(np.random.normal(0.0, np.sqrt(dt)))  # Wiener increment
        f_curr = (-(z_curr - zeta_bar) / tau_zeta).astype(np.float32)
        z_pred = z_curr + f_curr * dt + sigma_zeta * dW
        f_pred = (-(z_pred - zeta_bar) / tau_zeta).astype(np.float32)
        zeta[i] = z_curr + 0.5 * dt * (f_curr + f_pred) + sigma_zeta * dW

    return zeta

# -----------------------------
# Full network simulation
# -----------------------------

@jit(nopython=True, cache=False)
def simulate_network_numba(W_S, W_A, initial_condition, n_steps, dt, tau, r_m, beta, x_r,
                           model_type, zeta_array):
    N = initial_condition.size
    u = np.empty((n_steps, N), dtype=np.float32)
    u[0, :] = initial_condition
    for i in range(1, n_steps):
        zeta_curr = zeta_array[i - 1]
        zeta_next = zeta_array[i] if i < n_steps else zeta_curr
        u[i, :] = rk4_step_numba(u[i - 1, :], dt, W_S, W_A, tau, zeta_curr, zeta_next,
                                 model_type, r_m, beta, x_r)
    return u

# -----------------------------
# Pattern overlaps
# -----------------------------

@jit(nopython=True, cache=True, parallel=True)
def calculate_pattern_overlaps_numba(u, patterns, phi_params, g_params, use_g=True):
    n_timepoints, n_neurons = u.shape
    n_patterns = patterns.shape[0]
    overlaps = np.zeros((n_timepoints, n_patterns), dtype=np.float32)
    phi_patterns = np.zeros_like(patterns, dtype=np.float32)
    g_phi_patterns = np.zeros_like(patterns, dtype=np.float32)

    for p in range(n_patterns):
        phi_patterns[p, :] = apply_activation_sigmoid(patterns[p, :], phi_params[0], phi_params[1], phi_params[2])
        g_phi_patterns[p, :] = step_function_numba(phi_patterns[p, :], g_params[0], g_params[1])

    for t in prange(n_timepoints):
        r = apply_activation_sigmoid(u[t, :], phi_params[0], phi_params[1], phi_params[2])
        for p in range(n_patterns):
            g_phi_eta = g_phi_patterns[p, :]
            phi_eta = phi_patterns[p, :]
            var_g_phi_eta = np.var(g_phi_eta)
            var_r = np.var(r)
            var_phi_eta = np.var(phi_eta)
            if use_g:
                overlaps[t, p] = np.mean((g_phi_eta - np.mean(g_phi_eta)) * (r - np.mean(r))) / np.sqrt(var_g_phi_eta * var_r) if var_g_phi_eta > 1e-10 and var_r > 1e-10 else 0.0
            else:
                overlaps[t, p] = np.cov(phi_eta, r)[0, 1] / np.sqrt(var_phi_eta * var_r) if var_phi_eta > 1e-10 and var_r > 1e-10 else 0.0
    return overlaps

# -----------------------------
# Utilities
# -----------------------------

def get_numba_performance_info():
    import numba
    return {
        'numba_version': numba.__version__,
        'threading_layer': numba.config.THREADING_LAYER,
        'parallel_support': numba.config.NUMBA_NUM_THREADS,
        'cuda_available': numba.cuda.is_available() if hasattr(numba, 'cuda') else False
    }

def estimate_memory_usage(N, n_steps):
    u_memory = n_steps * N * 4
    W_memory = 2 * N * N * 4
    zeta_memory = n_steps * 4
    total_gb = (u_memory + W_memory + zeta_memory) / (1024**3)
    return {
        'total_gb': total_gb,
        'u_array_gb': u_memory / (1024**3),
        'connectivity_gb': W_memory / (1024**3),
        'recommended_max_neurons': int(np.sqrt(8 * 1024**3 / 4))
    }
