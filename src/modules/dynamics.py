"""
Dynamics module completo: Heun stocastico + OU, optional numba backend.
All arrays float32.
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.activation import sigmoid_function, relu_function, step_function

# ---------------------------
# Ornstein-Uhlenbeck Heun per mezzo-passo
# ---------------------------
def _ou_heun_step(z, dt_step, tau_zeta, zeta_bar, sigma_zeta, dW):
    g = np.sqrt(np.float32(2.0)) * np.float32(sigma_zeta) / np.sqrt(np.float32(tau_zeta))
    f_prev = (-(z - zeta_bar) / tau_zeta).astype(np.float32)
    z_pred = z + f_prev * dt_step + g * dW
    f_pred = (-(z_pred - zeta_bar) / tau_zeta).astype(np.float32)
    z_new = z + 0.5 * dt_step * (f_prev + f_pred) + g * dW
    return z_new.astype(np.float32)

# ---------------------------
# Initial conditions
# ---------------------------
def initial_condition_creator(init_cond_type, N, p=0, eta=None, pattern_idx=None, noise_level=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pattern_index = np.random.randint(0, p) if pattern_idx is None else pattern_idx
    if init_cond_type == "Random":
        initial_condition = np.random.normal(0.0, 0.1, N)
    elif init_cond_type == "Zero":
        initial_condition = np.zeros(N)
    elif init_cond_type == "Memory Pattern":
        initial_condition = eta[pattern_index].copy() if p > 0 and eta is not None else np.random.normal(0.0, 0.1, N)
    elif init_cond_type == "Negative Memory Pattern":
        initial_condition = -eta[pattern_index].copy() if p > 0 and eta is not None else np.random.normal(0.0, 0.1, N)
    else:  # Near Memory Pattern
        if p > 0 and eta is not None:
            pattern = eta[pattern_index % p]
            noise = np.random.normal(0.0, noise_level * np.std(pattern), N)
            initial_condition = pattern + noise
        else:
            initial_condition = np.random.normal(0.0, 0.1, N)
    return initial_condition.astype(np.float32)

# ---------------------------
# Network dynamics
# ---------------------------
def network_dynamics_recanatesi(u, W_S, W_A, activation_fn, tau, zeta_value=1.0):
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)
    phi_u = activation_fn(u).astype(np.float32)
    du_dt = (-u + W_S @ phi_u + zeta_value * (W_A @ phi_u)) / tau
    return du_dt.astype(np.float32)

def network_dynamics_brunel(u, W_S, W_A, activation_fn, tau, zeta_value=1.0):
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)
    total_input = W_S @ u + zeta_value * (W_A @ u)
    du_dt = (-u + activation_fn(total_input)) / tau
    return du_dt.astype(np.float32)

def network_dynamics(u, W_S, W_A, activation_fn, tau, zeta_value=1.0, model_type="recanatesi"):
    if model_type == "recanatesi":
        return network_dynamics_recanatesi(u, W_S, W_A, activation_fn, tau, zeta_value)
    elif model_type == "brunel":
        return network_dynamics_brunel(u, W_S, W_A, activation_fn, tau, zeta_value)
    else:
        raise ValueError("Unknown model_type. Choose 'recanatesi' or 'brunel'.")

# ---------------------------
# Simulate network: Heun stocastico + optional numba
# ---------------------------
def simulate_network(W_S,
                     W_A,
                     t_span,
                     dt,
                     tau,
                     initial_condition=None,
                     use_ou=False,
                     ou_params=None,
                     r_m=1.0,
                     beta=0.0,
                     x_r=0.0,
                     model_type="recanatesi",
                     constant_zeta=1.0,
                     use_numba=False,
                     seed=None):
    if use_numba:
        try:
            from modules.numba_dynamics import simulate_network_numba
            return simulate_network_numba(W_S, W_A, t_span, dt, tau,
                                         initial_condition, use_ou, ou_params,
                                         r_m, beta, x_r, model_type,
                                         constant_zeta, seed)
        except Exception:
            # fallback Python
            pass

    # Python implementation
    if seed is not None:
        np.random.seed(seed)

    dt = np.float32(dt)
    tau = np.float32(tau)
    r_m = np.float32(r_m)
    beta = np.float32(beta)
    x_r = np.float32(x_r)
    W_S = np.ascontiguousarray(W_S, dtype=np.float32)
    W_A = np.ascontiguousarray(W_A, dtype=np.float32)
    N = W_S.shape[0]

    if initial_condition is None:
        u = np.random.normal(0.0, 0.1, N).astype(np.float32)
    else:
        u = initial_condition.astype(np.float32, copy=False)

    def activation_fn(x):
        return sigmoid_function(x.astype(np.float32), r_m=r_m, beta=beta, x_r=x_r).astype(np.float32)

    t = np.arange(t_span[0], t_span[1], dt, dtype=np.float32)
    n_steps = len(t)
    u_hist = np.zeros((n_steps, N), dtype=np.float32)
    u_hist[0, :] = u.copy()
    zeta_array = np.zeros(n_steps, dtype=np.float32)

    if not use_ou:
        zeta_array[:] = np.float32(constant_zeta)
    else:
        tau_zeta = np.float32(ou_params['tau_zeta'])
        zeta_bar = np.float32(ou_params['zeta_bar'])
        sigma_zeta = np.float32(ou_params['sigma_zeta'])
        zeta_array[0] = zeta_bar
        half_dt = 0.5 * dt
        sqrt_half_dt = np.sqrt(half_dt).astype(np.float32)

    for i in range(1, n_steps):
        if use_ou:
            dW1 = np.float32(np.random.normal(0.0, 1.0) * sqrt_half_dt)
            dW2 = np.float32(np.random.normal(0.0, 1.0) * sqrt_half_dt)
            z_n = zeta_array[i-1]
            z_mid = _ou_heun_step(z_n, half_dt, tau_zeta, zeta_bar, sigma_zeta, dW1)
            z_next = _ou_heun_step(z_mid, half_dt, tau_zeta, zeta_bar, sigma_zeta, dW2)
            zeta_array[i] = z_next
            zeta_n = z_n
        else:
            zeta_n = zeta_array[i-1]
            z_next = zeta_n
            zeta_array[i] = z_next

        # Heun per u
        k1 = network_dynamics(u, W_S, W_A, activation_fn, tau, zeta_value=zeta_n, model_type=model_type)
        u_pred = u + dt * k1
        k2 = network_dynamics(u_pred, W_S, W_A, activation_fn, tau, zeta_value=z_next, model_type=model_type)
        u = u + 0.5 * dt * (k1 + k2)
        u = u.astype(np.float32)
        u_hist[i, :] = u

    return t, u_hist, zeta_array

# ---------------------------
# Overlap calculation
# ---------------------------
def calculate_pattern_overlaps(u, patterns, phi_params, g_params, use_g=True, use_numba=False):
    """
    Compute overlaps: corr( g(phi(pattern)), phi(state) ) for each time and pattern.
    Uses sigmoid_function and step_function. step_function expects q_f and x_f (as in your code).
    """
    if use_numba:
        try:
            from modules.numba_dynamics import calculate_pattern_overlaps_numba
            return calculate_pattern_overlaps_numba(u, patterns, phi_params, g_params, use_g=use_g)
        except Exception:
            pass  # fallback Python

    u = u.astype(np.float32, copy=False)
    patterns = patterns.astype(np.float32, copy=False)
    n_timepoints, n_neurons = u.shape
    n_patterns = patterns.shape[0]

    overlaps = np.zeros((n_timepoints, n_patterns), dtype=np.float32)
    phi_patterns = np.zeros_like(patterns, dtype=np.float32)
    g_phi_patterns = np.zeros_like(patterns, dtype=np.float32)

    for p in range(n_patterns):
        phi_patterns[p, :] = sigmoid_function(patterns[p, :], **phi_params).astype(np.float32)
        g_phi_patterns[p, :] = step_function(phi_patterns[p, :], **g_params).astype(np.float32)

    for ti in range(n_timepoints):
        r = sigmoid_function(u[ti, :], **phi_params).astype(np.float32)
        var_r = np.var(r)
        if var_r <= 1e-12:
            continue
        for p in range(n_patterns):
            a = g_phi_patterns[p] if use_g else phi_patterns[p]
            var_a = np.var(a)
            if var_a <= 1e-12:
                overlaps[ti, p] = 0.0
                continue
            cov = np.cov(a, r)[0, 1]
            overlaps[ti, p] = cov / np.sqrt(var_a * var_r)
    return overlaps.astype(np.float32)

# ---------------------------
# Plotting
# ---------------------------
def plot_network_dynamics(t, u, zeta, neuron_indices=None, max_display=10,
                          patterns=None, phi_params=None, g_params=None):
    t = t.astype(np.float32)
    u = u.astype(np.float32)
    is_zeta_array = isinstance(zeta, np.ndarray)
    N = u.shape[1]
    if neuron_indices is None:
        neuron_indices = list(range(min(N, max_display)))
    has_patterns = patterns is not None and len(patterns) > 0
    n_plots = 1 + int(is_zeta_array) + int(has_patterns)

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots),
                             gridspec_kw={'height_ratios': [3] + [1] * (n_plots - 1)})
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    ax1 = axes[0]
    ax2 = axes[1] if n_plots >= 2 else None
    ax3 = axes[2] if n_plots >= 3 else None

    for idx in neuron_indices:
        ax1.plot(t, u[:, idx], label=f'Neuron {idx}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neural Activity')
    ax1.set_title('Network Dynamics Simulation')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    if is_zeta_array and ax2 is not None:
        ax2.plot(t, zeta, 'r-', label='ζ(t)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('ζ(t)')
        ax2.set_title('Ornstein-Uhlenbeck Process')
        ax2.grid(True)

    if has_patterns and ax3 is not None:
        overlaps = calculate_pattern_overlaps(u, patterns, phi_params, g_params)
        for p in range(overlaps.shape[1]):
            ax3.plot(t, overlaps[:, p], label=f'Pattern {p}')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Overlap (Pearson r)')
        ax3.set_title('Memory Pattern Overlaps')
        ax3.set_ylim(-1.05, 1.05)
        ax3.legend(loc='upper right')
        ax3.grid(True)

    plt.tight_layout()
    return fig
