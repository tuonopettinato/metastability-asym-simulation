"""
modules/dynamics.py contains functions for simulating neural network dynamics using
different models and activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import N, use

# Import activation functions from connectivity module
from modules.activation import sigmoid_function, relu_function, step_function

# ================================================================
# Ornstein-Uhlenbeck process simulation
# ================================================================
def simulate_ou_process(t_span, dt, tau_zeta, zeta_bar, sigma_zeta, seed=None):
    """Simulate an Ornstein-Uhlenbeck process (Euler-Maruyama method, float32).

    Parameters
    ----------
    t_span : tuple
        Time span for simulation (t_start, t_end)
    dt : float
        Time step
    tau_zeta : float
        Time constant for OU process
    zeta_bar : float
        Mean value of OU process
    sigma_zeta : float
        Noise intensity
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    t : ndarray (float32)
        Time points
    zeta : ndarray (float32)
        OU process values at each time point
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Convert all parameters to float32 once and for all ---
    dt = np.float32(dt)
    tau_zeta = np.float32(tau_zeta)
    zeta_bar = np.float32(zeta_bar)
    sigma_zeta = np.float32(sigma_zeta)

    # Generate time points
    t = np.arange(t_span[0], t_span[1], dt, dtype=np.float32)
    n_steps = len(t)

    # Initialize zeta array
    zeta = np.zeros(n_steps, dtype=np.float32)
    zeta[0] = zeta_bar  # Start at mean value

    # Noise scaling factor (for Euler-Maruyama)
    noise_scale = np.sqrt(np.float32(2.0) * sigma_zeta**2 * tau_zeta * dt)

    # Gaussian white noise
    dW = np.random.normal(0, 1, n_steps).astype(np.float32)

    # Simulate OU process
    for i in range(1, n_steps):
        dzeta = ((-zeta[i - 1] + zeta_bar) * dt / tau_zeta
                 + noise_scale * dW[i] / tau_zeta)
        zeta[i] = zeta[i - 1] + dzeta
        # Alternative reflection (commented):
        # zeta[i] = zeta[i] if zeta[i] > 0 else -zeta[i]

    return t, zeta

def initial_condition_creator(init_cond_type, N, p=0, eta=None, pattern_idx=None, noise_level=0.5, seed = None):
    """Create initial condition based on specified type."""
    # Set up initial condition with proper noise calculation
    if seed is not None:
        np.random.seed(seed)
    pattern_index = np.random.randint(0, p) if pattern_idx is None else pattern_idx
    if init_cond_type == "Random":
        initial_condition = np.random.normal(0, 0.1, N)
    elif init_cond_type == "Zero":
        initial_condition = np.zeros(N)
    elif init_cond_type == "Memory Pattern":
        if p > 0:
            pattern = eta[pattern_index]
            initial_condition = pattern.copy()
        else:
            initial_condition = np.random.normal(0, 0.1, N)
    elif init_cond_type == "Negative Memory Pattern":
        if p > 0:
            pattern = eta[pattern_index]
            initial_condition = -pattern.copy()
        else:
            initial_condition = np.random.normal(0, 0.1, N)
    else:  # Near Memory Pattern
        if p > 0:
            pattern = eta[pattern_index % p]  # Getting the pattern based on index
            # Add noise scaled relative to pattern magnitude
            pattern_std = np.std(pattern)
            noise = np.random.normal(0, noise_level * pattern_std, N)
            initial_condition = pattern + noise
        else:
            initial_condition = np.random.normal(0, 0.1, N)
    return initial_condition.astype(np.float32)


# ================================================================
# Neural network dynamics (Recanatesi & Mazzucato model)
# ================================================================
def network_dynamics_recanatesi(t,
                                u,
                                W_S,
                                W_A,
                                activation_fn,
                                tau,
                                zeta_value=1.0):
    """Compute the right-hand side of the neural network dynamics equation
    using the Recanatesi and Mazzucato model:

    tau * du_i/dt = -u_i + sum_j W^S_ij * phi(u_j) + zeta(t) * sum_j W^A_ij * phi(u_j)

    Parameters
    ----------
    t : float
        Current time (not used for autonomous systems, but required by integrator)
    u : ndarray (float32)
        Current state of all neurons
    W_S : ndarray (float32)
        Symmetric connectivity matrix
    W_A : ndarray (float32)
        Asymmetric connectivity matrix
    activation_fn : function
        Activation function to apply to neural states
    tau : float
        Time constant for neural dynamics
    zeta_value : float
        Value of zeta(t) at current time (scalar or time-dependent)

    Returns
    -------
    du_dt : ndarray (float32)
        Rate of change of neural states
    """
    # --- Ensure parameters are float32 ---
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)

    # Apply activation function to all neurons
    phi_u = activation_fn(u).astype(np.float32)

    # Compute the symmetric and asymmetric contributions
    symmetric_input = (W_S @ phi_u).astype(np.float32)
    asymmetric_input = (W_A @ phi_u).astype(np.float32)

    # Dynamics equation
    du_dt = (-u + symmetric_input + zeta_value * asymmetric_input) / tau
    return du_dt.astype(np.float32)


# ================================================================
# Neural network dynamics (Brunel & Pereira model)
# ================================================================
def network_dynamics_brunel(t,
                            u,
                            W_S,
                            W_A,
                            activation_fn,
                            tau,
                            zeta_value=1.0):
    """Compute the right-hand side of the neural network dynamics equation
    using the Brunel and Pereira model:

    tau * du_i/dt = -u_i + phi(sum_{j≠i} W^S_ij * u_j + zeta(t) * sum_{j≠i} W^A_ij * u_j)

    Parameters
    ----------
    t : float
        Current time (not used for autonomous systems, but required by integrator)
    u : ndarray (float32)
        Current state of all neurons
    W_S : ndarray (float32)
        Symmetric connectivity matrix
    W_A : ndarray (float32)
        Asymmetric connectivity matrix
    activation_fn : function
        Activation function to apply to neural inputs
    tau : float
        Time constant for neural dynamics
    zeta_value : float
        Value of zeta(t) at current time (scalar or time-dependent)

    Returns
    -------
    du_dt : ndarray (float32)
        Rate of change of neural states
    """
    # --- Ensure parameters are float32 ---
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)

    # Compute total synaptic input
    symmetric_input = (W_S @ u).astype(np.float32)
    asymmetric_input = (W_A @ u).astype(np.float32)
    total_input = symmetric_input + zeta_value * asymmetric_input

    # Apply activation function to the total input
    activated_input = activation_fn(total_input).astype(np.float32)

    # Dynamics equation
    du_dt = (-u + activated_input) / tau
    return du_dt.astype(np.float32)
# ==


# ================================================================
# Wrapper which selects the dynamics model
# ================================================================
def network_dynamics(t,
                     u,
                     W_S,
                     W_A,
                     activation_fn,
                     tau,
                     zeta_value=1.0,
                     model_type="recanatesi"):
    """Wrapper function to select the appropriate dynamics model.
    
    Parameters
    ----------
    t : float
        Current time (not used for autonomous systems, but required by solve_ivp)
    u : ndarray (float32)
        Current state of all neurons
    W_S : ndarray (float32)
        Symmetric connectivity matrix
    W_A : ndarray (float32)
        Asymmetric connectivity matrix
    activation_fn : function
        Activation function to apply to neural states
    tau : float
        Time constant for neural dynamics
    zeta_value : float
        Value of zeta(t) at current time (scalar or time-dependent)
    model_type : str
        Which dynamics model to use: "recanatesi" or "brunel"
    
    Returns
    -------
    du_dt : ndarray (float32)
        Rate of change of neural states
    """
    # --- Convert to float32 once ---
    tau = np.float32(tau)
    zeta_value = np.float32(zeta_value)
    u = u.astype(np.float32, copy=False)
    W_S = W_S.astype(np.float32, copy=False)
    W_A = W_A.astype(np.float32, copy=False)

    if model_type == "recanatesi":
        return network_dynamics_recanatesi(t, u, W_S, W_A, activation_fn, tau,
                                           zeta_value)
    elif model_type == "brunel":
        return network_dynamics_brunel(t, u, W_S, W_A, activation_fn, tau,
                                       zeta_value)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose 'recanatesi' or 'brunel'."
        )


# ================================================================
# Complete simulation of the neural network
# ================================================================
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
                     use_numba=True):
    """Simulate neural network dynamics
    
    Parameters
    ----------
    W_S : ndarray (float32)
        Symmetric connectivity matrix
    W_A : ndarray (float32)
        Asymmetric connectivity matrix
    t_span : tuple
        Time span for simulation (t_start, t_end)
    dt : float
        Time step for simulation output
    tau : float
        Time constant for neural dynamics
    activation_param : float
        Activation parameter (beta for sigmoid, ignored for relu)
    initial_condition : ndarray or None
        Initial condition for all neurons. If None, small random values are used.
    use_ou : bool
        Whether to use Ornstein-Uhlenbeck process for zeta(t)
    ou_params : dict or None
        Parameters for OU process (tau_zeta, zeta_bar, sigma_zeta)
    r_m : float
        Maximum firing rate parameter for sigmoid function
    beta : float
        Inflection point for sigmoid function
    x_r : float
        Threshold parameter for sigmoid function
    model_type : str
        Which dynamics model to use: "recanatesi" or "brunel"
    constant_zeta : float or None
        Constant value for zeta when OU process is not used. If None, defaults to 1.0
    use_numba : bool
        Whether to use Numba-optimized functions for large networks
    
    Returns
    -------
    t : ndarray (float32)
        Time points
    u : ndarray (float32)
        Neural states at each time point, shape (n_timepoints, n_neurons)
    zeta : ndarray (float32) or float32
        OU process values at each time point or constant value
    """
    # --- Convert base parameters to float32 ---
    u = np.ascontiguousarray(initial_condition, dtype=np.float32)
    W_S = np.ascontiguousarray(W_S, dtype=np.float32)
    W_A = np.ascontiguousarray(W_A, dtype=np.float32)
    t_span = (np.float32(t_span[0]), np.float32(t_span[1]))
    dt = np.float32(dt)
    tau = np.float32(tau)
    r_m = np.float32(r_m)
    x_r = np.float32(x_r)
    beta = np.float32(beta)
    constant_zeta = (np.float32(constant_zeta)) if constant_zeta is not None else np.float32(1.0)

    # Ensure matrices are float32
    W_S = W_S.astype(np.float32, copy=False)
    W_A = W_A.astype(np.float32, copy=False)

    N = W_S.shape[0]  # Number of neurons

    # Set initial condition if not provided
    if initial_condition is None:
        initial_condition = np.random.normal(0, 0.1, N).astype(np.float32)
    else:
        initial_condition = initial_condition.astype(np.float32, copy=False)

    # Use Numba optimization for large networks (N > 1000) when enabled
    if use_numba and N > 1000:
        print(f"Using Numba optimization for {N} neurons...")
        try:
            from modules.numba_dynamics import (
                simulate_network_numba, simulate_ou_process_numba,
                get_numba_performance_info, estimate_memory_usage
            )
            
            # Print performance info
            perf_info = get_numba_performance_info()
            print(f"Numba version: {perf_info['numba_version']}")
            
            # Estimate memory usage
            mem_info = estimate_memory_usage(N, int((t_span[1] - t_span[0]) / dt))
            print(f"Estimated memory usage: {mem_info['total_gb']:.2f} GB")
            if mem_info['total_gb'] > 16:
                print("Warning: High memory usage detected. Consider reducing simulation time or using smaller networks.")

            return _simulate_network_numba_wrapper(W_S, W_A, t_span, dt, tau,
                                           initial_condition, use_ou,
                                           ou_params, r_m, beta, x_r, model_type, constant_zeta)
        except ImportError:
            print("Numba not available, falling back to standard simulation...")
            use_numba = False

    # Define activation function (sigmoid)
    def activation_fn(x):
        return sigmoid_function(x.astype(np.float32),
                                r_m=r_m,
                                beta=beta,
                                x_r=x_r).astype(np.float32)

    # Generate time points (float32)
    t = np.arange(t_span[0], t_span[1], dt, dtype=np.float32)
    n_steps = len(t)

    # Initialize state array (n_timepoints, n_neurons, float32)
    u = np.zeros((n_steps, N), dtype=np.float32)
    u[0, :] = initial_condition

    # Handle zeta(t) based on whether OU process is used
    if use_ou:
        if ou_params is None:
            raise ValueError("ou_params must be provided when use_ou=True")

        # Simulate OU process in float32
        _, zeta_array = simulate_ou_process(t_span=t_span,
                                            dt=dt,
                                            tau_zeta=ou_params['tau_zeta'],
                                            zeta_bar=ou_params['zeta_bar'],
                                            sigma_zeta=ou_params['sigma_zeta'],
                                            seed=ou_params.get('seed', None))
        zeta_array = zeta_array.astype(np.float32)
        zeta_output = zeta_array
    else:
        zeta_value = np.float32(constant_zeta if constant_zeta is not None else 1.0)
        zeta_array = np.full(n_steps, zeta_value, dtype=np.float32)
        zeta_output = zeta_value

    # Single RK4 integration loop
    for i in range(1, n_steps):
        # Get zeta values for this step
        zeta_curr = zeta_array[i-1]
        zeta_next = zeta_array[i] if i < len(zeta_array) else zeta_curr
        zeta_mid = np.float32(0.5) * (zeta_curr + zeta_next)
        
        # RK4 step (all float32)
        k1 = dt * network_dynamics(t[i-1], u[i-1], W_S, W_A, activation_fn, 
                                   tau, zeta_value=zeta_curr, model_type=model_type)
        k2 = dt * network_dynamics(t[i-1] + np.float32(0.5)*dt, u[i-1] + np.float32(0.5)*k1,
                                   W_S, W_A, activation_fn, tau, zeta_value=zeta_mid, model_type=model_type)
        k3 = dt * network_dynamics(t[i-1] + np.float32(0.5)*dt, u[i-1] + np.float32(0.5)*k2,
                                   W_S, W_A, activation_fn, tau, zeta_value=zeta_mid, model_type=model_type)
        k4 = dt * network_dynamics(t[i-1] + dt, u[i-1] + k3,
                                   W_S, W_A, activation_fn, tau, zeta_value=zeta_next, model_type=model_type)
        
        u[i] = u[i-1] + (k1 + np.float32(2.0)*k2 + np.float32(2.0)*k3 + k4) / np.float32(6.0)

    return t, u, zeta_output

# ================================================================
# Wrapper function for Numba-optimized simulations
# ================================================================

def _simulate_network_numba_wrapper(W_S, W_A, t_span, dt, tau, initial_condition, use_ou,
                                   ou_params, r_m, beta, x_r, model_type, constant_zeta):
    """Wrapper function to call Numba-optimized simulation"""
    from modules.numba_dynamics import simulate_network_numba, simulate_ou_process_numba

    # --- Convert inputs to float32 once ---
    dt = np.float32(dt)
    tau = np.float32(tau)
    r_m = np.float32(r_m)
    beta = np.float32(beta)
    x_r = np.float32(x_r)
    W_S = W_S.astype(np.float32, copy=False)
    W_A = W_A.astype(np.float32, copy=False)
    initial_condition = initial_condition.astype(np.float32, copy=False)

    # Generate time points
    t = np.arange(np.float32(t_span[0]), np.float32(t_span[1]), dt, dtype=np.float32)
    n_steps = len(t)

    # Handle zeta(t) based on whether OU process is used
    if use_ou:
        if ou_params is None:
            raise ValueError("ou_params must be provided when use_ou=True")
        
        zeta_array = simulate_ou_process_numba(
            n_steps,
            dt,
            np.float32(ou_params['tau_zeta']),
            np.float32(ou_params['zeta_bar']),
            np.float32(ou_params['sigma_zeta'])
        ).astype(np.float32)
        zeta_output = zeta_array
    else:
        zeta_value = np.float32(constant_zeta if constant_zeta is not None else 1.0)
        zeta_array = np.full(n_steps, zeta_value, dtype=np.float32)
        zeta_output = zeta_value

    # Map model type to integer for Numba
    model_type_int = 0 if model_type == 'recanatesi' else 1

    # Run Numba-optimized simulation
    u = simulate_network_numba(
        W_S, W_A, initial_condition, n_steps, dt, tau, r_m, beta, x_r,
        model_type_int, zeta_array
    ).astype(np.float32)

    return t, u, zeta_output

# ================================================================
# Calculate overlaps between network states and memory patterns
# ================================================================

def calculate_pattern_overlaps(u, patterns, phi_params, g_params, use_numba=True, use_g=True):
    """Calculate covariance-based overlaps between network states and memory patterns
    
    Overlap definition: cov(g(φ(η)), r) / sqrt(var(g(φ(η))) * var(r))
    where:
    - η is the memory pattern
    - r = φ(state) is the firing rate of the current state
    - g is the connectivity function applied to patterns
    - φ is the activation function
    
    This computes the Pearson correlation coefficient between g(φ(η)) and φ(state).
    
    Parameters
    ----------
    u : ndarray
        Neural states, shape (n_timepoints, n_neurons)
    patterns : ndarray
        Memory patterns, shape (n_patterns, n_neurons)
    phi_params : dict
        Parameters for phi function (r_m, beta, x_r for sigmoid; amplitude for relu)
    g_params : dict
        Parameters for g function
    use_numba : bool
        Whether to use Numba optimization for large networks
    use_g : bool
        Whether to apply the g function to patterns (default: True)
    
    Returns
    -------
    overlaps : ndarray (float32)
        Overlaps (covariance-based correlation), shape (n_timepoints, n_patterns)
        Range: [-1, 1] where 1 = perfect correlation, -1 = perfect anti-correlation, 0 = uncorrelated
    """
    # --- Convert arrays to float32 once ---
    u = u.astype(np.float32, copy=False)
    patterns = patterns.astype(np.float32, copy=False)
    
    n_timepoints, n_neurons = u.shape
    n_patterns = patterns.shape[0]

    if phi_params is None:
        raise ValueError("phi_params cannot be None")
    if g_params is None:
        raise ValueError("g_params cannot be None")

    # Use Numba optimization if enabled and network is large
    if use_numba and (n_neurons > 1000 or n_timepoints > 5000):
        try:
            from modules.numba_dynamics import calculate_pattern_overlaps_numba
            
            phi_params_array = np.array([
                phi_params.get('r_m', 1.0),
                phi_params.get('beta', 1.0), 
                phi_params.get('x_r', 0.0),
            ], dtype=np.float32)
            
            g_params_array = np.array([
                g_params.get('q', 1.0),
                g_params.get('x', 0.0)
            ], dtype=np.float32)
        
            
            return calculate_pattern_overlaps_numba(u, patterns, phi_params_array, g_params_array,
                                                    use_g=use_g)
        except ImportError:
            print("Numba not available for overlap calculation, using standard method...")

    # --- Validate phi_params and g_params ---
    if 'r_m' not in phi_params or 'beta' not in phi_params or 'x_r' not in phi_params:
        raise ValueError("Missing required phi_params for sigmoid: ['r_m', 'beta', 'x_r']")
    else: None  # all good
    if 'q' not in g_params or 'x' not in g_params:
        raise ValueError("Missing required g_params for step: ['q', 'x']")
    else: None  # all good

    # Initialize overlaps array (float32)
    overlaps = np.zeros((n_timepoints, n_patterns), dtype=np.float32)

    # Pre-compute φ(η) for all patterns
    phi_patterns = np.zeros_like(patterns, dtype=np.float32)
    for p in range(n_patterns):
        phi_patterns[p, :] = sigmoid_function(patterns[p, :].astype(np.float32),
                                                r_m=np.float32(phi_params['r_m']),
                                                beta=np.float32(phi_params['beta']),
                                                x_r=np.float32(phi_params['x_r'])).astype(np.float32)

    # Pre-compute g(φ(η)) for all patterns
    g_phi_patterns = np.zeros_like(phi_patterns, dtype=np.float32)
    for p in range(n_patterns):
        g_phi_patterns[p, :] = step_function(phi_patterns[p, :].astype(np.float32),
                                                 q_f=np.float32(g_params['q']),
                                                 x_f=np.float32(g_params['x'])).astype(np.float32)

    # Compute overlaps for each timepoint and pattern
    for t_idx in range(n_timepoints):
        state = u[t_idx, :]
        r = sigmoid_function(state.astype(np.float32),
                                r_m=np.float32(phi_params['r_m']),
                                beta=np.float32(phi_params['beta']),
                                x_r=np.float32(phi_params['x_r'])).astype(np.float32)

        for p in range(n_patterns):
            g_phi_eta = g_phi_patterns[p, :]
            phi_eta = phi_patterns[p, :]

            var_g_phi_eta = np.var(g_phi_eta)
            var_phi_eta = np.var(phi_eta)
            var_r = np.var(r)

            if use_g:
                if var_g_phi_eta > 1e-10 and var_r > 1e-10:
                    cov = np.cov(g_phi_eta, r)[0, 1]
                    overlaps[t_idx, p] = cov / np.sqrt(var_g_phi_eta * var_r)
                else:
                    overlaps[t_idx, p] = 0.0
            else:
                if var_phi_eta > 1e-10 and var_r > 1e-10:
                    cov = np.cov(phi_eta, r)[0, 1]
                    overlaps[t_idx, p] = cov / np.sqrt(var_phi_eta * var_r)
                else:
                    overlaps[t_idx, p] = 0.0

    return overlaps.astype(np.float32)

# ================================================================
# Plotting functions
# ================================================================

def plot_network_dynamics(t,
                          u,
                          zeta,
                          neuron_indices=None,
                          max_display=10,
                          patterns=None,
                          phi_params=None,
                          g_params=None):
    """Plot network dynamics results
    
    Parameters
    ----------
    t : ndarray
        Time points
    u : ndarray
        Neural states, shape (n_timepoints, n_neurons)
    zeta : ndarray or float
        OU process values at each time point or constant
    neuron_indices : list or None
        Indices of neurons to plot. If None, a subset is selected.
    max_display : int
        Maximum number of neurons to display
    patterns : ndarray or None
        Memory patterns, shape (n_patterns, n_neurons)
    phi_params : dict
        Parameters for phi function
    g_params : dict
        Parameters for g function
    
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    # Convert arrays to float32
    t = t.astype(np.float32)
    u = u.astype(np.float32)

    if isinstance(zeta, np.ndarray):
        zeta = zeta.astype(np.float32)

    N = u.shape[1]  # Number of neurons

    # Select neuron indices if not provided
    if neuron_indices is None:
        if N <= max_display:
            neuron_indices = list(range(N))
        else:
            step = max(1, N // max_display)
            neuron_indices = list(range(0, N, step))[:max_display]

    # Determine number of subplots
    is_zeta_array = isinstance(zeta, np.ndarray)
    has_patterns = patterns is not None and len(patterns) > 0
    n_plots = 1 + int(is_zeta_array) + int(has_patterns)

    # Initialize figure and axes
    if n_plots > 1:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots),
                                 gridspec_kw={'height_ratios': [3] + [1]*(n_plots-1)})
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        ax1 = axes[0]
        ax2 = axes[1] if n_plots >= 2 else None
        ax3 = axes[2] if n_plots >= 3 else None
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = ax3 = None

    # Plot neural activity
    for i, idx in enumerate(neuron_indices):
        ax1.plot(t, u[:, idx], label=f'Neuron {idx+1}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neural Activity')
    ax1.set_title('Network Dynamics Simulation')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot zeta(t) if array
    if is_zeta_array and ax2 is not None:
        ax2.plot(t, zeta, 'r-', label='ζ(t)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('ζ(t)')
        ax2.set_title('Ornstein-Uhlenbeck Process')
        ax2.grid(True)

    # Plot pattern overlaps if provided
    if has_patterns:
        overlaps = calculate_pattern_overlaps(u, patterns, phi_params, g_params)
        ax_overlap = ax3 if n_plots == 3 else ax2 if n_plots == 2 else None
        if ax_overlap is not None:
            for p in range(overlaps.shape[1]):
                ax_overlap.plot(t, overlaps[:, p], label=f'Pattern {p+1}')
            ax_overlap.set_xlabel('Time')
            ax_overlap.set_ylabel('Overlap (Pearson r)')
            ax_overlap.set_title('Memory Pattern Overlaps')
            ax_overlap.set_ylim(-1.05, 1.05)
            ax_overlap.legend(loc='upper right')
            ax_overlap.grid(True)

    plt.tight_layout()
    return fig
