"""
modules/dynamics.py contains functions for simulating neural network dynamics using different models and activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import activation functions from connectivity module
from modules.activation import tanh_function, sigmoid_function, relu_function, threshold_function


# Ornstein-Uhlenbeck process simulation
def simulate_ou_process(t_span, dt, tau_zeta, zeta_bar, sigma_zeta, seed=None):
    """Simulate an Ornstein-Uhlenbeck process
    
    Parameters:
    -----------
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
    
    Returns:
    --------
    t : ndarray
        Time points
    zeta : ndarray
        OU process values at each time point
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate time points
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)

    # Initialize zeta array
    zeta = np.zeros(n_steps)
    zeta[0] = zeta_bar  # Start at mean value

    # Noise scaling factor
    noise_scale = np.sqrt(2 * sigma_zeta**2 * tau_zeta * dt)

    # Simulate OU process using Euler-Maruyama
    for i in range(1, n_steps):
        # Gaussian white noise
        dW = np.random.normal(0, 1)

        # Update zeta using OU dynamics
        dzeta = (-zeta[i - 1] +
                 zeta_bar) * dt / tau_zeta + noise_scale * dW / tau_zeta
        zeta[i] = zeta[i - 1] + dzeta

    return t, zeta


# Neural network dynamics - computing the rate of change of neural states for both models
def network_dynamics_recanatesi(t,
                                u,
                                W_S,
                                W_A,
                                activation_fn,
                                tau,
                                zeta_value=1.0):
    """Compute the right-hand side of the neural network dynamics equation
    using the Recanatesi and Mazzucato model:
    
    tau * du_i/dt = -u_i + sum_j W_S_ij * phi(u_j) + zeta(t) * sum_j W_A_ij * phi(u_j)
    
    Parameters:
    -----------
    t : float
        Current time (not used for autonomous systems, but required by solve_ivp)
    u : ndarray
        Current state of all neurons
    W_S : ndarray
        Symmetric connectivity matrix
    W_A : ndarray
        Asymmetric connectivity matrix
    activation_fn : function
        Activation function to apply to neural states
    tau : float
        Time constant for neural dynamics
    zeta_value : float or ndarray
        Value of zeta(t) at current time (scalar or time-dependent)
    
    Returns:
    --------
    du_dt : ndarray
        Rate of change of neural states
    """
    # Apply activation function to all neurons
    phi_u = activation_fn(u)

    # Compute the symmetric and asymmetric contributions
    symmetric_input = W_S @ phi_u
    asymmetric_input = W_A @ phi_u

    du_dt = (-u + symmetric_input + zeta_value * asymmetric_input) / tau

    return du_dt


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
    
    Parameters:
    -----------
    t : float
        Current time (not used for autonomous systems, but required by solve_ivp)
    u : ndarray
        Current state of all neurons
    W_S : ndarray
        Symmetric connectivity matrix
    W_A : ndarray
        Asymmetric connectivity matrix
    activation_fn : function
        Activation function to apply to neural inputs
    tau : float
        Time constant for neural dynamics
    zeta_value : float or ndarray
        Value of zeta(t) at current time (scalar or time-dependent)
    
    Returns:
    --------
    du_dt : ndarray
        Rate of change of neural states
    """

    symmetric_input = W_S @ u
    asymmetric_input = W_A @ u

    # Apply the activation function to the total input
    total_input = symmetric_input + zeta_value * asymmetric_input
    activated_input = activation_fn(total_input)

    du_dt = (-u + activated_input) / tau

    return du_dt


def network_dynamics(t,
                     u,
                     W_S,
                     W_A,
                     activation_fn,
                     tau,
                     zeta_value=1.0,
                     model_type="recanatesi"):
    """Wrapper function to select the appropriate dynamics model.
    
    Parameters:
    -----------
    t : float
        Current time (not used for autonomous systems, but required by solve_ivp)
    u : ndarray
        Current state of all neurons
    W_S : ndarray
        Symmetric connectivity matrix
    W_A : ndarray
        Asymmetric connectivity matrix
    activation_fn : function
        Activation function to apply to neural states
    tau : float
        Time constant for neural dynamics
    zeta_value : float or ndarray
        Value of zeta(t) at current time (scalar or time-dependent)
    model_type : str
        Which dynamics model to use: "recanatesi" or "brunel"
    
    Returns:
    --------
    du_dt : ndarray
        Rate of change of neural states
    """
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


def simulate_network(W_S,
                     W_A,
                     t_span,
                     dt,
                     tau,
                     activation_type,
                     activation_param,
                     initial_condition=None,
                     use_ou=False,
                     ou_params=None,
                     r_m=1.0,
                     theta=0.0,
                     x_r=0.0,
                     model_type="recanatesi",
                     constant_zeta=None,
                     use_numba=True):
    """Simulate neural network dynamics
    
    Parameters:
    -----------
    W_S : ndarray
        Symmetric connectivity matrix
    W_A : ndarray
        Asymmetric connectivity matrix
    t_span : tuple
        Time span for simulation (t_start, t_end)
    dt : float
        Time step for simulation output
    tau : float
        Time constant for neural dynamics
    activation_type : str
        Type of activation function ('sigmoid', 'relu')
    activation_param : float
    initial_condition : ndarray or None
        Initial condition for all neurons. If None, small random values are used.
    use_ou : bool
        Whether to use Ornstein-Uhlenbeck process for zeta(t)
    ou_params : dict or None
        Parameters for OU process (tau_zeta, zeta_bar, sigma_zeta)
    r_m : float
        Maximum firing rate parameter for sigmoid function
    theta : float
        Inflection point for sigmoid function
    x_r : float
        Threshold parameter for sigmoid function
    model_type : str
        Which dynamics model to use: "recanatesi" or "brunel"
    constant_zeta : float or None
        Constant value for zeta when OU process is not used. If None, defaults to 1.0
    use_numba : bool
        Whether to use Numba-optimized functions for large networks (default: True)
    
    Returns:
    --------
    t : ndarray
        Time points
    u : ndarray
        Neural states at each time point, shape (n_timepoints, n_neurons)
    zeta : ndarray or float
        OU process values at each time point or constant value
    """
    N = W_S.shape[0]  # Number of neurons

    # Set initial condition if not provided
    if initial_condition is None:
        initial_condition = np.random.normal(0, 0.1, N)

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
            
            return _simulate_network_numba_wrapper(W_S, W_A, t_span, dt, tau, activation_type,
                                                  activation_param, initial_condition, use_ou,
                                                  ou_params, r_m, x_r, model_type, constant_zeta)
        except ImportError:
            print("Numba not available, falling back to standard simulation...")
            use_numba = False

    # Choose activation function with parameters using imported functions from activation.py
    if activation_type == 'sigmoid':
        # For sigmoid, use all sigmoid parameters (r_m, beta, x_r)
        # activation_param is beta (steepness)
        def activation_fn(x):
            return sigmoid_function(x, r_m=r_m, beta=activation_param, x_r=x_r)
    elif activation_type == 'relu':
        # For ReLU, use standard ReLU with no clipping
        def activation_fn(x):
            return relu_function(x, amplitude=1.0)
    else: # error 
        raise ValueError(f"Unknown activation type: {activation_type}. Choose 'sigmoid' or 'relu'.")

    # Generate time points
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)

    # Initialize state array (n_timepoints, n_neurons)
    u = np.zeros((n_steps, N))
    u[0, :] = initial_condition  # Set initial condition

    # Handle zeta(t) based on whether OU process is used
    if use_ou:
        # Ensure ou_params are provided
        if ou_params is None:
            raise ValueError("ou_params must be provided when use_ou=True")

        # Simulate OU process
        _, zeta_array = simulate_ou_process(t_span=t_span,
                                      dt=dt,
                                      tau_zeta=ou_params['tau_zeta'],
                                      zeta_bar=ou_params['zeta_bar'],
                                      sigma_zeta=ou_params['sigma_zeta'],
                                      seed=ou_params.get('seed', None))

        zeta_output = zeta_array

    else:
        # Use constant zeta value (default to 1.0 if not provided)
        zeta_value = constant_zeta if constant_zeta is not None else 1.0
        zeta_array = np.full(n_steps, zeta_value)
        zeta_output = zeta_value

    # Single RK4 integration loop
    for i in range(1, n_steps):
        # Get zeta values for this step
        zeta_curr = zeta_array[i-1]
        zeta_next = zeta_array[i] if i < len(zeta_array) else zeta_curr
        zeta_mid = 0.5 * (zeta_curr + zeta_next)
        
        # RK4 step
        k1 = dt * network_dynamics(t[i-1], u[i-1], W_S, W_A, activation_fn, 
                                tau, zeta_value=zeta_curr, model_type=model_type)
        k2 = dt * network_dynamics(t[i-1] + 0.5*dt, u[i-1] + 0.5*k1, W_S, W_A, 
                                activation_fn, tau, zeta_value=zeta_mid, model_type=model_type)
        k3 = dt * network_dynamics(t[i-1] + 0.5*dt, u[i-1] + 0.5*k2, W_S, W_A, 
                                activation_fn, tau, zeta_value=zeta_mid, model_type=model_type)
        k4 = dt * network_dynamics(t[i-1] + dt, u[i-1] + k3, W_S, W_A, activation_fn, 
                                tau, zeta_value=zeta_next, model_type=model_type)
        
        u[i] = u[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6.0


    return t, u, zeta_output


def _simulate_network_numba_wrapper(W_S, W_A, t_span, dt, tau, activation_type,
                                   activation_param, initial_condition, use_ou,
                                   ou_params, r_m, x_r, model_type, constant_zeta):
    """Wrapper function to call Numba-optimized simulation"""
    from modules.numba_dynamics import simulate_network_numba, simulate_ou_process_numba
    
    # Generate time points
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)
    
    # Handle zeta(t) based on whether OU process is used
    if use_ou:
        if ou_params is None:
            raise ValueError("ou_params must be provided when use_ou=True")
        
        zeta_array = simulate_ou_process_numba(n_steps, dt,
                                              ou_params['tau_zeta'],
                                              ou_params['zeta_bar'],
                                              ou_params['sigma_zeta'])
        zeta_output = zeta_array
    else:
        zeta_value = constant_zeta if constant_zeta is not None else 1.0
        zeta_array = np.full(n_steps, zeta_value)
        zeta_output = zeta_value
    
    # Convert string parameters to integers for Numba
    activation_type_int = 0 if activation_type == 'sigmoid' else 1
    model_type_int = 0 if model_type == 'recanatesi' else 1
    
    # Set activation parameters
    amplitude = 1.0 if activation_type == 'relu' else activation_param
    
    # Run Numba-optimized simulation
    u = simulate_network_numba(W_S, W_A, initial_condition, n_steps, dt, tau,
                              activation_type_int, r_m, activation_param, x_r, amplitude,
                              model_type_int, zeta_array)
    
    return t, u, zeta_output

def calculate_pattern_overlaps(u, patterns, phi_function_type, phi_params, g_function_type, g_params, use_numba=True):
    """Calculate covariance-based overlaps between network states and memory patterns
    
    Overlap definition: cov(g(φ(η)), r) / sqrt(var(g(φ(η))) * var(r))
    where:
    - η is the memory pattern
    - r = φ(state) is the firing rate of the current state
    - g is the connectivity function applied to patterns
    - φ is the activation function
    
    This computes the Pearson correlation coefficient between g(φ(η)) and φ(state).
    
    Parameters:
    -----------
    u : ndarray
        Neural states, shape (n_timepoints, n_neurons)
    patterns : ndarray
        Memory patterns, shape (n_patterns, n_neurons)
    phi_function_type : str
        Type of phi activation function ('sigmoid' or 'relu')
    phi_params : dict
        Parameters for phi function (r_m, beta, x_r for sigmoid; amplitude for relu)
    g_function_type : str
        Type of g function ('sigmoid' or 'step')
    g_params : dict
        Parameters for g function
    use_numba : bool
        Whether to use Numba optimization for large networks
    
    Returns:
    --------
    overlaps : ndarray
        Overlaps (covariance-based correlation), shape (n_timepoints, n_patterns)
        Range: [-1, 1] where 1 = perfect correlation, -1 = perfect anti-correlation, 0 = uncorrelated
    """
    n_timepoints, n_neurons = u.shape
    n_patterns = patterns.shape[0]

    # Validate that parameters are provided (they are now required arguments)
    if phi_params is None:
        raise ValueError("phi_params cannot be None")
    if g_params is None:
        raise ValueError("g_params cannot be None")

    # Use Numba optimization for large networks
    if use_numba and (n_neurons > 1000 or n_timepoints > 5000):
        try:
            from modules.numba_dynamics import calculate_pattern_overlaps_numba
            
            # Convert parameters to arrays for Numba
            phi_params_array = np.array([
                phi_params.get('r_m', 1.0),
                phi_params.get('beta', 1.0), 
                phi_params.get('x_r', 0.0),
                phi_params.get('amplitude', 1.0)
            ])
            
            g_params_array = np.array([
                g_params.get('r_m', 1.0),
                g_params.get('beta', 1.0),
                g_params.get('x_r', 0.0),
                g_params.get('q', 1.0),
                g_params.get('x', 0.0)
            ])
            
            phi_type_int = 0 if phi_function_type == 'sigmoid' else 1
            g_type_int = 0 if g_function_type == 'sigmoid' else 1
            
            return calculate_pattern_overlaps_numba(u, patterns, phi_params_array, g_params_array,
                                                   phi_type_int, g_type_int)
        except ImportError:
            print("Numba not available for overlap calculation, using standard method...")
    
    # Check required parameters for phi function
    if phi_function_type == 'sigmoid':
        required_phi = ['r_m', 'beta', 'x_r']
        missing_phi = [p for p in required_phi if p not in phi_params]
        if missing_phi:
            raise ValueError(f"Missing required phi_params for sigmoid: {missing_phi}")
    elif phi_function_type == 'relu':
        if 'amplitude' not in phi_params:
            raise ValueError("Missing required phi_params for relu: ['amplitude']")
    
    # Check required parameters for g function
    if g_function_type == 'sigmoid':
        required_g = ['r_m', 'beta', 'x_r']
        missing_g = [p for p in required_g if p not in g_params]
        if missing_g:
            raise ValueError(f"Missing required g_params for sigmoid: {missing_g}")
    elif g_function_type == 'step':
        required_g = ['q', 'x']
        missing_g = [p for p in required_g if p not in g_params]
        if missing_g:
            raise ValueError(f"Missing required g_params for step: {missing_g}")

    # Initialize overlaps array
    overlaps = np.zeros((n_timepoints, n_patterns))

    # Pre-compute φ(η) for all patterns
    phi_patterns = np.zeros_like(patterns)
    for p in range(n_patterns):
        if phi_function_type == 'sigmoid':
            phi_patterns[p, :] = sigmoid_function(patterns[p, :], 
                                                 r_m=phi_params['r_m'],
                                                 beta=phi_params['beta'],
                                                 x_r=phi_params['x_r'])
        else:  # relu
            phi_patterns[p, :] = relu_function(patterns[p, :], 
                                              amplitude=phi_params['amplitude'])

    # Pre-compute g(φ(η)) for all patterns
    g_phi_patterns = np.zeros_like(phi_patterns)
    for p in range(n_patterns):
        if g_function_type == 'sigmoid':
            g_phi_patterns[p, :] = sigmoid_function(phi_patterns[p, :],
                                                   r_m=g_params['r_m'],
                                                   beta=g_params['beta'],
                                                   x_r=g_params['x_r'])
        else:  # step
            from modules.activation import step_function
            g_phi_patterns[p, :] = step_function(phi_patterns[p, :],
                                                q_f=g_params['q'],
                                                x_f=g_params['x'])

    # Calculate overlaps for each timepoint and pattern
    for t in range(n_timepoints):
        # Compute r = φ(state) for current timepoint
        state = u[t, :]
        if phi_function_type == 'sigmoid':
            r = sigmoid_function(state,
                               r_m=phi_params['r_m'],
                               beta=phi_params['beta'],
                               x_r=phi_params['x_r'])
        else:  # relu
            r = relu_function(state, amplitude=phi_params['amplitude'])

        for p in range(n_patterns):
            # Get g(φ(η)) for pattern p
            g_phi_eta = g_phi_patterns[p, :]
            
            # Calculate variances
            var_g_phi_eta = np.var(g_phi_eta)
            var_r = np.var(r)
            
            if var_g_phi_eta > 1e-10 and var_r > 1e-10:
                # Calculate covariance between g(φ(η)) and r
                cov_g_phi_eta_r = np.cov(g_phi_eta, r)[0, 1]
                
                # Calculate overlap using corrected correlation definition
                overlaps[t, p] = cov_g_phi_eta_r / np.sqrt(var_g_phi_eta * var_r)
            else:
                overlaps[t, p] = 0.0

    return overlaps

def calculte_normalized_dot_product(u, patterns):
    """Calculate normalized dot product (cosine similarity) overlaps between network states and memory patterns
    
    This is more appropriate for neural network dynamics than Pearson correlation as it:
    - Preserves magnitude information
    - Matches the mathematical structure of Hopfield-type networks
    - Doesn't assume zero-mean data
    
    Parameters:
    -----------
    u : ndarray
        Neural states, shape (n_timepoints, n_neurons)
    patterns : ndarray
        Memory patterns, shape (n_patterns, n_neurons)
    
    Returns:
    --------
    overlaps : ndarray
        Overlaps (normalized dot product), shape (n_timepoints, n_patterns)
        Range: [-1, 1] where 1 = perfect match, -1 = perfect anti-match, 0 = orthogonal
    """
    n_timepoints, n_neurons = u.shape
    n_patterns = patterns.shape[0]

    # Initialize overlaps array
    overlaps = np.zeros((n_timepoints, n_patterns))

    # Calculate normalized dot product for each timepoint and pattern
    for t in range(n_timepoints):
        for p in range(n_patterns):
            # Extract the state at timepoint t and pattern p
            state = u[t, :]
            pattern = patterns[p, :]

            # Calculate norms
            state_norm = np.linalg.norm(state)
            pattern_norm = np.linalg.norm(pattern)

            if state_norm > 0 and pattern_norm > 0:
                # Calculate normalized dot product (cosine similarity)
                overlaps[t, p] = np.dot(state, pattern) / (state_norm * pattern_norm)
            else:
                overlaps[t, p] = 0.0

    return overlaps


def plot_network_dynamics(t,
                          u,
                          zeta,
                          neuron_indices=None,
                          max_display=10,
                          patterns=None,
                          phi_function_type='sigmoid',
                          phi_params=None,
                          g_function_type='step',
                          g_params=None):
    """Plot network dynamics results
    
    Parameters:
    -----------
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
    phi_function_type : str
        Type of phi activation function
    phi_params : dict
        Parameters for phi function
    g_function_type : str
        Type of g function
    g_params : dict
        Parameters for g function
    
    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    """
    N = u.shape[1]  # Number of neurons

    # If specific neuron indices aren't provided, choose some
    if neuron_indices is None:
        if N <= max_display:
            neuron_indices = list(range(N))
        else:
            # Choose neurons to display (evenly spaced)
            step = max(1, N // max_display)
            neuron_indices = list(range(0, N, step))[:max_display]

    # Calculate the number of subplots needed
    is_zeta_array = isinstance(zeta, np.ndarray)
    has_patterns = patterns is not None and len(patterns) > 0

    n_plots = 1  # At least one plot for neural activity
    if is_zeta_array:
        n_plots += 1
    if has_patterns:
        n_plots += 1

    # Initialize variables
    ax1 = None
    ax2 = None
    ax3 = None
    
    # Create figure with appropriate number of subplots
    if n_plots > 1:
        fig, axes = plt.subplots(
            n_plots,
            1,
            figsize=(10, 4 * n_plots),
            gridspec_kw={'height_ratios': [3] + [1] * (n_plots - 1)})
        # Ensure axes is always a list for consistent indexing
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        ax1 = axes[0]
        if n_plots >= 2:
            ax2 = axes[1]
        if n_plots >= 3:
            ax3 = axes[2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot neural activity
    for i, idx in enumerate(neuron_indices):
        ax1.plot(t, u[:, idx], label=f'Neuron {idx+1}')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neural Activity')
    ax1.set_title('Network Dynamics Simulation')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot zeta if time-dependent
    if is_zeta_array:
        if ax2 is not None:
            ax2.plot(t, zeta, 'r-', label='ζ(t)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('ζ(t)')
            ax2.set_title('Ornstein-Uhlenbeck Process')
            ax2.grid(True)

    # Plot pattern overlaps if patterns provided
    if has_patterns:
        # Calculate overlaps 
        overlaps = calculate_pattern_overlaps(u, patterns, phi_function_type, phi_params, g_function_type, g_params)

        # Plot the overlaps
        ax_overlap = ax3 if n_plots == 3 else ax2 if n_plots == 2 else None
        if ax_overlap is not None:
            for p in range(overlaps.shape[1]):
                ax_overlap.plot(t, overlaps[:, p], label=f'Pattern {p+1}')

            ax_overlap.set_xlabel('Time')
            ax_overlap.set_ylabel('Overlap (Pearson r)')
            ax_overlap.set_title('Memory Pattern Overlaps')
            ax_overlap.set_ylim(
                -1.05, 1.05)  #  correlations range from -1 to 1
            ax_overlap.legend(loc='upper right')
            ax_overlap.grid(True)

    plt.tight_layout()
    return fig