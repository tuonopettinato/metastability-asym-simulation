"""
Numba-optimized neural network dynamics with stochastic Heun integration.
Matches dynamics.py exactly: no interpolation of zeta.
All arrays and computations float32.
"""

import numpy as np
from numba import njit, prange

# -----------------------------
# Activation functions
# -----------------------------
@njit(cache=True)
def sigmoid_numba(x, r_m=1.0, beta=1.0, x_r=0.0):
    x = np.float32(x)
    r_m = np.float32(r_m)
    beta = np.float32(beta)
    x_r = np.float32(x_r)
    val = -beta * (x - x_r)
    if val > np.float32(500.0): val = np.float32(500.0)
    elif val < np.float32(-500.0): val = np.float32(-500.0)
    return r_m / (np.float32(1.0) + np.exp(val))

@njit(cache=True)
def relu_numba(x, amplitude=1.0):
    x = np.float32(x)
    amplitude = np.float32(amplitude)
    return amplitude * x if x > 0.0 else np.float32(0.0)

@njit(cache=True)
def step_function_numba(x, q_f=1.0, x_f=0.0):
    x = np.float32(x)
    q_f = np.float32(q_f)
    x_f = np.float32(x_f)
    return q_f if x > x_f else -(np.float32(1.0) - q_f)

@njit(cache=True, parallel=True)
def apply_activation_sigmoid(u, r_m, beta, x_r):
    N = u.size
    result = np.empty(N, dtype=np.float32)
    for i in prange(N):
        result[i] = sigmoid_numba(u[i], r_m, beta, x_r)
    return result

# -----------------------------
# Matrix-vector multiply
# -----------------------------
@njit(cache=True, parallel=True)
def matrix_vector_multiply_parallel(W, v):
    N = W.shape[0]
    result = np.zeros(N, dtype=np.float32)
    for i in prange(N):
        for j in range(N):
            result[i] += np.float32(W[i, j]) * np.float32(v[j])
    return result

# -----------------------------
# Network dynamics
# -----------------------------
@njit(cache=True)
def network_dynamics_recanatesi(u, W_S, W_A, tau, zeta_value, r_m, beta, x_r):
    phi_u = apply_activation_sigmoid(u, r_m, beta, x_r)
    symmetric_input = matrix_vector_multiply_parallel(W_S, phi_u)
    asymmetric_input = matrix_vector_multiply_parallel(W_A, phi_u)
    N = u.size
    du_dt = np.empty(N, dtype=np.float32)
    for i in range(N):
        du_dt[i] = (-u[i] + symmetric_input[i] + zeta_value * asymmetric_input[i]) / tau
    return du_dt

@njit(cache=True)
def network_dynamics_brunel(u, W_S, W_A, tau, zeta_value, r_m, beta, x_r):
    symmetric_input = matrix_vector_multiply_parallel(W_S, u)
    asymmetric_input = matrix_vector_multiply_parallel(W_A, u)
    total_input = symmetric_input + zeta_value * asymmetric_input
    activated_input = apply_activation_sigmoid(total_input, r_m, beta, x_r)
    du_dt = (-u + activated_input) / tau
    return du_dt

# -----------------------------
# Stochastic Heun step (single-step)
# -----------------------------
@njit(cache=True)
def heun_step(u, dt, W_S, W_A, tau, zeta_value, model_type, r_m, beta, x_r, dW=None):
    du1 = network_dynamics_recanatesi(u, W_S, W_A, tau, zeta_value, r_m, beta, x_r) if model_type == 0 else \
          network_dynamics_brunel(u, W_S, W_A, tau, zeta_value, r_m, beta, x_r)
    u_predict = u + dt * du1
    du2 = network_dynamics_recanatesi(u_predict, W_S, W_A, tau, zeta_value, r_m, beta, x_r) if model_type == 0 else \
          network_dynamics_brunel(u_predict, W_S, W_A, tau, zeta_value, r_m, beta, x_r)
    return u + 0.5 * dt * (du1 + du2)

@njit(cache=True)
def ou_heun_step(z, dt_step, tau_zeta, zeta_bar, sigma_zeta, dW):
    g = np.sqrt(np.float32(2.0)) * sigma_zeta / np.sqrt(tau_zeta)
    f_prev = -(z - zeta_bar) / tau_zeta
    z_pred = z + f_prev * dt_step + g * dW
    f_pred = -(z_pred - zeta_bar) / tau_zeta
    z_new = z + 0.5 * dt_step * (f_prev + f_pred) + g * dW
    return z_new

# -----------------------------
# Full network simulation (Heun single-step)
# -----------------------------
@njit(cache=True)
def simulate_network_numba(W_S, W_A, initial_condition, n_steps, dt, tau,
                           r_m, beta, x_r, model_type, use_ou,
                           tau_zeta=1.0, zeta_bar=1.0, sigma_zeta=0.0,
                           constant_zeta=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = initial_condition.size
    u_hist = np.empty((n_steps, N), dtype=np.float32)
    u = initial_condition.astype(np.float32).copy()
    u_hist[0, :] = u

    zeta_array = np.empty(n_steps, dtype=np.float32)
    if not use_ou:
        zeta_array[:] = np.float32(constant_zeta)
    else:
        zeta_array[0] = np.float32(zeta_bar)

    for i in range(1, n_steps):
        if use_ou:
            dW = np.float32(np.random.normal(0.0, 1.0) * np.sqrt(dt))
            z_n = zeta_array[i-1]
            z_next = ou_heun_step(z_n, dt, tau_zeta, zeta_bar, sigma_zeta, dW)
            zeta_array[i] = z_next
            zeta_n = z_n
        else:
            zeta_n = np.float32(constant_zeta)
            z_next = zeta_n
            zeta_array[i] = z_next

        u = heun_step(u, dt, W_S, W_A, tau, zeta_n, model_type, r_m, beta, x_r)
        u_hist[i, :] = u

    return u_hist, zeta_array

# -----------------------------
# Pattern overlaps
# -----------------------------
@njit(cache=True, parallel=True)
def calculate_pattern_overlaps_numba(u_hist, patterns, phi_params, g_params, use_g=True):
    n_steps, N = u_hist.shape
    n_patterns = patterns.shape[0]

    r_m_phi, beta_phi, x_r_phi = np.float32(phi_params[0]), np.float32(phi_params[1]), np.float32(phi_params[2])
    q_f, x_f = np.float32(g_params[0]), np.float32(g_params[1])

    phi_patterns = np.empty_like(patterns, dtype=np.float32)
    g_phi_patterns = np.empty_like(patterns, dtype=np.float32)

    for p in prange(n_patterns):
        for i in range(N):
            phi_patterns[p, i] = sigmoid_numba(patterns[p, i], r_m_phi, beta_phi, x_r_phi)
            g_phi_patterns[p, i] = q_f if phi_patterns[p, i] > x_f else -(np.float32(1.0) - q_f)

    overlaps = np.zeros((n_steps, n_patterns), dtype=np.float32)

    for t_idx in prange(n_steps):
        r = np.empty(N, dtype=np.float32)
        for i in range(N):
            r[i] = sigmoid_numba(u_hist[t_idx, i], r_m_phi, beta_phi, x_r_phi)

        mean_r = np.mean(r)
        var_r = np.var(r)
        if var_r < 1e-10: continue

        for p in range(n_patterns):
            a = g_phi_patterns[p] if use_g else phi_patterns[p]
            mean_a = np.mean(a)
            var_a = np.var(a)
            if var_a < 1e-10:
                overlaps[t_idx, p] = 0.0
                continue

            cov = np.mean((a - mean_a)*(r - mean_r))
            overlaps[t_idx, p] = cov / np.sqrt(var_a*var_r)

    return overlaps