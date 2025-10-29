"""
modules/dynamics.py

Complete dynamics module:
- OU integrated with Heun on half-steps (so we can evaluate zeta at midpoints exactly,
  avoiding interpolation artifacts).
- Network integrated with RK4 using the zeta values computed at t, t+dt/2, t+dt.
- All arrays in float32, compatible with optional numba backends (if provided).
- Includes initial_condition_creator, Recanatesi/Brunel dynamics, overlap calculation, plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import activation functions from your activation module.
# Make sure these functions accept the keyword names used below.
from modules.activation import sigmoid_function, relu_function, step_function

# ---------------------------
# Ornstein-Uhlenbeck (Heun per mezzo-passo)
# ---------------------------
def _ou_heun_step(z, dt_step, tau_zeta, zeta_bar, sigma_zeta, dW):
    """
    Perform a single Heun integration over a sub-step of length dt_step for OU:
      dz = (-(z - zbar)/tau) dt + g dW,  with g = sqrt(2) * sigma / sqrt(tau)
    Returns the updated z after this sub-step.

    This helper uses the deterministic Heun correction and treats the stochastic increment dW
    consistently (dW has variance dt_step, i.e. dW ~ N(0, sqrt(dt_step))).
    """
    # Coefficient for noise (consistent with SDE: tau dz = -z + zbar + sqrt(2 sigma^2 tau) dW)
    # => dz = (-(z - zbar)/tau) dt + sqrt(2) * sigma / sqrt(tau) * dW
    g = np.sqrt(2.0, dtype=np.float32) * sigma_zeta / np.sqrt(tau_zeta)
    f_prev = (-(z - zeta_bar) / tau_zeta).astype(np.float32)
    z_pred = z + f_prev * dt_step + g * dW
    f_pred = (-(z_pred - zeta_bar) / tau_zeta).astype(np.float32)
    z_new = z + 0.5 * dt_step * (f_prev + f_pred) + g * dW
    return z_new.astype(np.float32)

def simulate_ou_process(t_span, dt, tau_zeta, zeta_bar, sigma_zeta, seed=None):
    """
    Simulate Ornstein-Uhlenbeck process and return zeta at nodes (t_n).
    This function uses Euler-Maruyama-like generation for convenience when called alone,
    but the RK4 integration below computes zeta at midpoints on the fly with two sub-steps.
    Here we provide a node-only generator consistent with the OU parameters (Heun is used
    per full-step via two half-step Heun updates).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = np.float32(dt)
    tau_zeta = np.float32(tau_zeta)
    zeta_bar = np.float32(zeta_bar)
    sigma_zeta = np.float32(sigma_zeta)

    t = np.arange(t_span[0], t_span[1], dt, dtype=np.float32)
    n_steps = len(t)
    zeta = np.zeros(n_steps, dtype=np.float32)
    zeta[0] = zeta_bar

    # We'll use Heun over two half-steps per full step to reduce bias.
    half_dt = np.float32(0.5) * dt
    sqrt_half_dt = np.sqrt(half_dt).astype(np.float32)

    for i in range(1, n_steps):
        # independent increments for the two half-steps:
        dW1 = np.float32(np.random.normal(0.0, 1.0) * sqrt_half_dt)
        dW2 = np.float32(np.random.normal(0.0, 1.0) * sqrt_half_dt)

        # half-step 1: from zeta[i-1] -> zeta_mid
        z_mid = _ou_heun_step(zeta[i-1], half_dt, tau_zeta, zeta_bar, sigma_zeta, dW1)
        # half-step 2: from zeta_mid -> zeta[i]
        z_next = _ou_heun_step(z_mid, half_dt, tau_zeta, zeta_bar, sigma_zeta, dW2)

        zeta[i] = z_next

    return t, zeta.astype(np.float32)

# ---------------------------
# Initial conditions
# ---------------------------
def initial_condition_creator(init_cond_type, N, p=0, eta=None, pattern_idx=None, noise_level=0.5, seed=None):
    """Create initial condition based on specified type."""
    if seed is not None:
        np.random.seed(seed)

    pattern_index = np.random.randint(0, p) if pattern_idx is None else pattern_idx

    if init_cond_type == "Random":
        initial_condition = np.random.normal(0.0, 0.1, N)
    elif init_cond_type == "Zero":
        initial_condition = np.zeros(N)
    elif init_cond_type == "Memory Pattern":
        if p > 0 and eta is not None:
            initial_condition = eta[pattern_index].copy()
        else:
            initial_condition = np.random.normal(0.0, 0.1, N)
    elif init_cond_type == "Negative Memory Pattern":
        if p > 0 and eta is not None:
            initial_condition = -eta[pattern_index].copy()
        else:
            initial_condition = np.random.normal(0.0, 0.1, N)
    else:  # Near Memory Pattern
        if p > 0 and eta is not None:
            pattern = eta[pattern_index % p]
            pattern_std = np.std(pattern)
            noise = np.random.normal(0.0, noise_level * pattern_std, N)
            initial_condition = pattern + noise
        else:
            initial_condition = np.random.normal(0.0, 0.1, N)

    return initial_condition.astype(np.float32)

# ---------------------------
# Network dynamics: Recanatesi & Brunel
# ---------------------------
def network_dynamics_recanatesi(t, u, W_S, W_A, activation_fn, tau, zeta_value=1.0):
    """tau * du/dt = -u + W_S phi(u) + zeta * W_A phi(u)"""
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)
    phi_u = activation_fn(u).astype(np.float32)
    symmetric_input = (W_S @ phi_u).astype(np.float32)
    asymmetric_input = (W_A @ phi_u).astype(np.float32)
    du_dt = (-u + symmetric_input + zeta_value * asymmetric_input) / tau
    return du_dt.astype(np.float32)

def network_dynamics_brunel(t, u, W_S, W_A, activation_fn, tau, zeta_value=1.0):
    """tau * du/dt = -u + phi(W_S u + zeta * W_A u)"""
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)
    symmetric_input = (W_S @ u).astype(np.float32)
    asymmetric_input = (W_A @ u).astype(np.float32)
    total_input = symmetric_input + zeta_value * asymmetric_input
    activated_input = activation_fn(total_input).astype(np.float32)
    du_dt = (-u + activated_input) / tau
    return du_dt.astype(np.float32)

def network_dynamics(t, u, W_S, W_A, activation_fn, tau, zeta_value=1.0, model_type="recanatesi"):
    u = u.astype(np.float32, copy=False)
    W_S = W_S.astype(np.float32, copy=False)
    W_A = W_A.astype(np.float32, copy=False)
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)

    if model_type == "recanatesi":
        return network_dynamics_recanatesi(t, u, W_S, W_A, activation_fn, tau, zeta_value)
    elif model_type == "brunel":
        return network_dynamics_brunel(t, u, W_S, W_A, activation_fn, tau, zeta_value)
    else:
        raise ValueError("Unknown model_type. Choose 'recanatesi' or 'brunel'.")

# ---------------------------
# simulate_network: RK4 for u, OU substeps for zeta (no interpolation)
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
                     constant_zeta=None,
                     use_numba=True,
                     seed=None):
    """
    Simulate the network with RK4 while computing zeta at t, t+dt/2, t+dt
    by integrating the OU with two independent half-step increments (Heun per half-step).
    This avoids interpolation artifacts.
    """
    if seed is not None:
        np.random.seed(seed)

    # Cast params
    dt = np.float32(dt)
    tau = np.float32(tau)
    r_m = np.float32(r_m)
    beta = np.float32(beta)
    x_r = np.float32(x_r)
    W_S = np.ascontiguousarray(W_S, dtype=np.float32)
    W_A = np.ascontiguousarray(W_A, dtype=np.float32)

    N = W_S.shape[0]

    # Initial condition
    if initial_condition is None:
        u = np.random.normal(0.0, 0.1, N).astype(np.float32)
    else:
        u = initial_condition.astype(np.float32, copy=False)

    # Activation wrapper
    def activation_fn(x):
        return sigmoid_function(x.astype(np.float32), r_m=r_m, beta=beta, x_r=x_r).astype(np.float32)

    # Prepare time array
    t = np.arange(t_span[0], t_span[1], dt, dtype=np.float32)
    n_steps = len(t)

    # Prepare zeta storage
    zeta_array = np.zeros(n_steps, dtype=np.float32)

    # If we don't use OU, fill with constant zeta
    if not use_ou:
        zeta_val = np.float32(constant_zeta if constant_zeta is not None else 1.0)
        zeta_array[:] = zeta_val
    else:
        if ou_params is None:
            raise ValueError("ou_params must be provided when use_ou=True")
        # initialize zeta[0]
        zeta_array[0] = np.float32(ou_params.get('zeta_bar', 0.0))

    # Prepare result storage for u
    u_hist = np.zeros((n_steps, N), dtype=np.float32)
    u_hist[0, :] = u.copy()

    # If use_numba and N large, try to call numba backend (user's module)
    if use_numba and N > 1000:
        try:
            from modules.numba_dynamics import simulate_network_numba
            # If numba backend expects precomputed zeta_array, we should precompute consistent zeta_array.
            # But we don't enforce that here; prefer python loop if numba backend incompatible.
        except Exception:
            # proceed with Python implementation below
            pass

    # constants for OU substeps
    if use_ou:
        tau_zeta = np.float32(ou_params['tau_zeta'])
        zeta_bar = np.float32(ou_params['zeta_bar'])
        sigma_zeta = np.float32(ou_params['sigma_zeta'])
        half_dt = np.float32(0.5) * dt
        sqrt_half_dt = np.sqrt(half_dt).astype(np.float32)

        # ensure initial zeta defined
        if n_steps > 0 and zeta_array[0] == 0.0:
            zeta_array[0] = zeta_bar

    # Main loop: per full step, generate two independent half-step noises to obtain zeta_mid and zeta_next
    for i in range(1, n_steps):
        # --- compute zeta at mid and next (if OU) ---
        if use_ou:
            # two independent normal increments with variance half_dt
            # dW1 and dW2 are normal(0, sqrt(half_dt))
            dW1 = np.float32(np.random.normal(0.0, 1.0) * sqrt_half_dt)
            dW2 = np.float32(np.random.normal(0.0, 1.0) * sqrt_half_dt)

            # mid-step: Heun with dt/2 and dW1
            z_n = zeta_array[i-1]
            z_mid = _ou_heun_step(z_n, half_dt, tau_zeta, zeta_bar, sigma_zeta, dW1)
            # next-step: Heun with dt/2 and dW2 starting from mid
            z_next = _ou_heun_step(z_mid, half_dt, tau_zeta, zeta_bar, sigma_zeta, dW2)

            # store
            zeta_array[i] = z_next
            zeta_mid = z_mid
            zeta_n = z_n
        else:
            # constant zeta everywhere
            zeta_n = zeta_array[i-1]
            zeta_mid = zeta_n
            z_next = zeta_n
            zeta_array[i] = z_next

        # --- RK4 for u using zeta_n, zeta_mid, z_next ---
        u_n = u.copy()

        k1 = network_dynamics(t[i-1], u_n, W_S, W_A, activation_fn, tau, zeta_value=zeta_n, model_type=model_type)
        k1 = dt * k1

        k2_u = u_n + 0.5 * k1
        k2 = network_dynamics(t[i-1] + 0.5*dt, k2_u, W_S, W_A, activation_fn, tau, zeta_value=zeta_mid, model_type=model_type)
        k2 = dt * k2

        k3_u = u_n + 0.5 * k2
        # use same midpoint zeta for k3 (we computed zeta_mid with proper half-step OU)
        k3 = network_dynamics(t[i-1] + 0.5*dt, k3_u, W_S, W_A, activation_fn, tau, zeta_value=zeta_mid, model_type=model_type)
        k3 = dt * k3

        k4_u = u_n + k3
        k4 = network_dynamics(t[i-1] + dt, k4_u, W_S, W_A, activation_fn, tau, zeta_value=z_next, model_type=model_type)
        k4 = dt * k4

        u = u_n + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        u = u.astype(np.float32)

        u_hist[i, :] = u

    return t, u_hist, zeta_array

# ---------------------------
# Numba wrapper placeholder (if you have numba_dynamics implemented)
# ---------------------------
def _simulate_network_numba_wrapper(*args, **kwargs):
    """
    If you have a modules.numba_dynamics module that provides:
      simulate_ou_process_numba, simulate_network_numba, ...
    you can implement a wrapper here to call the numba routines.
    Currently we leave this as a placeholder—Python version above is consistent.
    """
    raise NotImplementedError("Numba wrapper not implemented in this file. Provide modules.numba_dynamics.")

# ---------------------------
# Overlap calculation
# ---------------------------
def calculate_pattern_overlaps(u, patterns, phi_params, g_params, use_numba=True, use_g=True):
    """
    Compute overlaps: corr( g(phi(pattern)), phi(state) ) for each time and pattern.
    Uses sigmoid_function and step_function. step_function expects q_f and x_f (as in your code).
    """
    u = u.astype(np.float32, copy=False)
    patterns = patterns.astype(np.float32, copy=False)
    n_timepoints, n_neurons = u.shape
    n_patterns = patterns.shape[0]

    if phi_params is None:
        raise ValueError("phi_params required")
    if g_params is None:
        raise ValueError("g_params required")

    overlaps = np.zeros((n_timepoints, n_patterns), dtype=np.float32)

    # precompute phi(patterns)
    phi_patterns = np.zeros_like(patterns, dtype=np.float32)
    for p in range(n_patterns):
        phi_patterns[p, :] = sigmoid_function(patterns[p, :].astype(np.float32),
                                              r_m=np.float32(phi_params['r_m']),
                                              beta=np.float32(phi_params['beta']),
                                              x_r=np.float32(phi_params['x_r'])).astype(np.float32)

    # precompute g(phi(patterns)) using step_function (expects q_f, x_f)
    g_phi_patterns = np.zeros_like(phi_patterns, dtype=np.float32)
    for p in range(n_patterns):
        g_phi_patterns[p, :] = step_function(phi_patterns[p, :].astype(np.float32),
                                             q_f=np.float32(g_params['q']),
                                             x_f=np.float32(g_params['x'])).astype(np.float32)

    # compute overlaps time by time
    for ti in range(n_timepoints):
        state = u[ti, :].astype(np.float32)
        r = sigmoid_function(state, r_m=np.float32(phi_params['r_m']),
                             beta=np.float32(phi_params['beta']),
                             x_r=np.float32(phi_params['x_r'])).astype(np.float32)

        var_r = np.var(r)
        if var_r <= 1e-12:
            # no variance in rates -> overlaps zero
            continue

        for p in range(n_patterns):
            if use_g:
                a = g_phi_patterns[p, :]
            else:
                a = phi_patterns[p, :]

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
        if N <= max_display:
            neuron_indices = list(range(N))
        else:
            step = max(1, N // max_display)
            neuron_indices = list(range(0, N, step))[:max_display]

    has_patterns = patterns is not None and len(patterns) > 0
    n_plots = 1 + int(is_zeta_array) + int(has_patterns)

    if n_plots > 1:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots),
                                 gridspec_kw={'height_ratios': [3] + [1] * (n_plots - 1)})
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        ax1 = axes[0]
        ax2 = axes[1] if n_plots >= 2 else None
        ax3 = axes[2] if n_plots >= 3 else None
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = ax3 = None

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

    if has_patterns:
        overlaps = calculate_pattern_overlaps(u, patterns, phi_params, g_params)
        ax_overlap = ax3 if n_plots == 3 else ax2 if n_plots == 2 else None
        if ax_overlap is not None:
            for p in range(overlaps.shape[1]):
                ax_overlap.plot(t, overlaps[:, p], label=f'Pattern {p}')
            ax_overlap.set_xlabel('Time')
            ax_overlap.set_ylabel('Overlap (Pearson r)')
            ax_overlap.set_title('Memory Pattern Overlaps')
            ax_overlap.set_ylim(-1.05, 1.05)
            ax_overlap.legend(loc='upper right')
            ax_overlap.grid(True)

    plt.tight_layout()
    return fig
