"""
connectivity.py contains functions to generate and analyze connectivity matrices
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from modules.activation import tanh_function, threshold_function, relu_function, sigmoid_function, step_function


def _enforce_correlation_constraint(eta_patterns, max_correlation, alpha, pattern_mean, pattern_sigma, max_iterations=100):
    """
    Iteratively refine patterns to ensure pairwise correlations stay below max_correlation.
    
    Parameters:
    -----------
    eta_patterns : ndarray
        Memory patterns of shape (p, N)
    max_correlation : float
        Maximum allowed correlation between patterns
    alpha : float
        Sparsity parameter for pattern refinement
    pattern_mean : float
        Mean for generating new random values
    pattern_sigma : float
        Standard deviation for generating new random values
    max_iterations : int
        Maximum number of refinement iterations
        
    Returns:
    --------
    eta_refined : ndarray
        Patterns with correlations below max_correlation
    """
    p, N = eta_patterns.shape
    eta_refined = eta_patterns.copy()
    
    for iteration in range(max_iterations):
        # Calculate all pairwise correlations
        max_corr_found = 0.0
        violating_pairs = []
        
        for i in range(p):
            for j in range(i + 1, p):
                # Calculate correlation, handling zero variance patterns
                pattern_i = eta_refined[i, :]
                pattern_j = eta_refined[j, :]
                
                if np.std(pattern_i) > 1e-10 and np.std(pattern_j) > 1e-10:
                    corr = np.corrcoef(pattern_i, pattern_j)[0, 1]
                    if not np.isnan(corr):
                        abs_corr = abs(corr)
                        max_corr_found = max(max_corr_found, abs_corr)
                        if abs_corr > max_correlation:
                            violating_pairs.append((i, j, abs_corr))
        
        # If all correlations are within bounds, we're done
        if max_corr_found <= max_correlation:
            break
            
        # Refine the most violating patterns
        if violating_pairs:
            # Sort by violation severity and fix the worst one
            violating_pairs.sort(key=lambda x: x[2], reverse=True)
            i, j, _ = violating_pairs[0]
            
            # Add small random perturbations to reduce correlation
            noise_strength = 0.1 * pattern_sigma
            
            # Perturb pattern j (keep pattern i as reference)
            random_noise = np.random.normal(0, noise_strength, N)
            eta_refined[j, :] += random_noise
            
            # Reapply sparsity constraint if alpha < 1.0
            if alpha < 1.0:
                sorted_indices = np.argsort(np.abs(eta_refined[j, :]))[::-1]
                n_active = int(alpha * N)
                inactive_indices = sorted_indices[n_active:]
                eta_refined[j, inactive_indices] = 0
    
    return eta_refined


def generate_connectivity_matrix(N,
                                 p,
                                 q,
                                 c,
                                 A_S,
                                 f_r_m=1.0,
                                 f_beta=1.0,
                                 f_x_r=0.0,
                                 f_type='sigmoid',
                                 f_q=1.0,
                                 f_x=0.0,
                                 g_r_m=1.0,
                                 g_beta=1.0,
                                 g_x_r=0.0,
                                 g_type='sigmoid',
                                 g_q=1.0,
                                 g_x=0.0,
                                 pattern_mean=0.0,
                                 pattern_sigma=1.0,
                                 apply_sigma_cutoff=True,
                                 phi_function_type='sigmoid',
                                 phi_amplitude=1.0,
                                 phi_beta=1.0,
                                 phi_r_m=1.0,
                                 phi_x_r=0.0,
                                 apply_phi_to_patterns=True,
                                 apply_er_to_asymmetric=False,
                                 alpha=1.0,
                                 enforce_max_correlation=False,
                                 max_correlation=0.5):
    """
    Generate connectivity matrices (symmetric, asymmetric, and total)
    
    Parameters:
    -----------
    N : int
        Number of neurons
    p : int
        Number of patterns for symmetric component
    q : int
        Number of patterns for asymmetric component (q <= p)
    c : float
        Connection probability (0-1) for Erdös-Rényi model
    A_S : float
        Amplitude parameter for symmetric component
    f_r_m : float
        Maximum firing rate for f sigmoid function
    f_beta : float
        Steepness parameter for f sigmoid function
    f_x_r : float
        Threshold parameter for f sigmoid function
    f_type : str
        Type of f function ('sigmoid' or 'step')
    f_q : float
        Step value for f step function
    f_x : float
        Step threshold for f step function
    g_r_m : float
        Maximum firing rate for g sigmoid function
    g_beta : float
        Steepness parameter for g sigmoid function
    g_x_r : float
        Threshold parameter for g sigmoid function
    g_type : str
        Type of g function ('sigmoid' or 'step')
    g_q : float
        Step value for g step function
    g_x : float
        Step threshold for g step function
    pattern_mean : float
        Mean of the Gaussian distribution for memory patterns
    pattern_sigma : float
        Standard deviation of the Gaussian distribution for memory patterns
    apply_sigma_cutoff : bool
        Whether to apply 1σ cutoff to patterns (values below μ + σ set to zero), default: True
    apply_phi_to_patterns : bool
        Whether to apply the activation function (φ) to the patterns, default: True
    apply_er_to_asymmetric : bool
        Whether to apply Erdős-Rényi connectivity to asymmetric component. 
        If True: Apply c_ij to both symmetric and asymmetric components.
        If False: Apply c_ij only to symmetric component, default: False
    alpha : float
        Memory pattern sparsity parameter (0-1). Controls the fraction of neurons 
        that are active (non-zero) in each memory pattern. alpha=1.0 means all 
        neurons can be active (dense patterns), alpha=0.1 means only 10% of neurons 
        are active per pattern (sparse patterns). Reduces correlation between memories.
    enforce_max_correlation : bool
        Switch to enable/disable correlation constraint enforcement. When True, 
        patterns are iteratively refined to ensure correlations stay below max_correlation.
    max_correlation : float
        Maximum allowed correlation between any pair of memory patterns (0-1). 
        Only used when enforce_max_correlation=True. Lower values create more 
        orthogonal memories but may require more iterations.
    
    Returns:
    --------
    W_S : ndarray
        Symmetric component of connectivity matrix
    W_A : ndarray
        Asymmetric component of connectivity matrix
    W : ndarray
        Total connectivity matrix (W_S + W_A)
    eta : ndarray
        Memory patterns used to generate the connectivity matrices
    """
    # Generate random memory patterns from Gaussian distribution
    eta_raw = np.random.normal(pattern_mean, pattern_sigma, size=(p, N))
    # eta_raw[eta_raw < 0] = 0 # firing rates are positive (but they are not firing rates)

    # Apply memory pattern sparsity (alpha parameter)
    if alpha < 1.0:
        # For each pattern, keep only top alpha fraction of neurons active
        for mu in range(p):
            # Get indices sorted by absolute value (descending)
            sorted_indices = np.argsort(np.abs(eta_raw[mu, :]))[::-1]
            # Calculate number of neurons to keep active
            n_active = int(alpha * N)
            # Set inactive neurons to zero
            inactive_indices = sorted_indices[n_active:]
            eta_raw[mu, inactive_indices] = 0

    # Conditionally apply 1-sigma cutoff (applied after sparsity)
    if apply_sigma_cutoff:
        # Cut all values below 1 sigma above the mean
        eta_raw[eta_raw < pattern_mean + pattern_sigma] = 0

    # Enforce maximum correlation constraint if enabled
    if enforce_max_correlation and p > 1:
        eta_raw = _enforce_correlation_constraint(eta_raw, max_correlation, alpha, pattern_mean, pattern_sigma)

    ###########################################################################
    #### -- Erdös-Rényi connection matrix c_ij -- #############################
    ###########################################################################
    c_ij_raw = np.random.binomial(1, c, size=(N, N))
    # Make symmetric by taking the upper triangle and its transpose
    c_ij = np.triu(c_ij_raw, 1)  # Upper triangle without diagonal
    c_ij = c_ij + c_ij.T  # Make symmetric!
    # Add diagonal entries with probability c
    np.fill_diagonal(c_ij, np.random.binomial(1, c, size=N))
    ###########################################################################

    # Calculate N_S (average number of connections per neuron)
    N_S = N * c

    # Set up f and g functions (for pattern interaction in connectivity matrix)
    # f and g can be either sigmoid or step functions with different parameters
    if f_type == 'sigmoid':
        f = lambda x: sigmoid_function(x, f_r_m, f_beta, f_x_r)
    else:  # step function
        f = lambda x: step_function(x, f_q, f_x)

    if g_type == 'sigmoid':
        g = lambda x: sigmoid_function(x, g_r_m, g_beta, g_x_r)
    else:  # step function
        g = lambda x: step_function(x, g_q, g_x)

    # Set up φ function
    if phi_function_type is None:
        phi_function_type = 'sigmoid'  # Default to sigmoid if not specified'

    if phi_function_type == 'tanh':
        phi = lambda x: tanh_function(x, phi_amplitude, phi_beta)
    elif phi_function_type == 'sigmoid':
        phi = lambda x: sigmoid_function(x, phi_r_m, phi_beta, phi_x_r)
    elif phi_function_type == 'relu':
        phi = lambda x: relu_function(x, phi_amplitude)
    else:  # threshold
        phi = lambda x: threshold_function(x, phi_amplitude)

    if apply_phi_to_patterns:
        eta = np.zeros_like(eta_raw)
        for mu in range(p):
            eta[mu, :] = phi(eta_raw[mu, :])
    else:
        eta = eta_raw

    # Initialize symmetric component
    W_S = np.zeros((N, N))

    # Calculate symmetric component
    for mu in range(p):
        # Apply f and g functions to memory patterns
        f_eta_i = f(eta[mu, :]).reshape(-1, 1)  # Column vector
        g_eta_j = g(eta[mu, :]).reshape(1, -1)  # Row vector

        # centering 
        f_eta_i -= np.mean(f_eta_i)
        g_eta_j -= np.mean(g_eta_j)
        
        # Compute the matrix element-wise product and ensure symmetry
        product_matrix = np.outer(f_eta_i, g_eta_j)
        # symmetric_product = 0.5 * (product_matrix + product_matrix.T)
        # it is already symmetric
        symmetric_product = product_matrix
        
        # Accumulate
        W_S += symmetric_product

    # Apply scaling and connection probability with improved scaling for metastability

    W_S = (c_ij * A_S / N_S) * W_S

    # Set diagonal elements to zero
    np.fill_diagonal(W_S, 0)

    # Ensure W_S is perfectly symmetric (due to numerical precision issues)
    # W_S = 0.5 * (W_S + W_S.T) # commenting this to allow imperfect asymmetry if f neq g

    # Initialize asymmetric component
    W_A = np.zeros((N, N))

    # Calculate asymmetric component (if q > 0)
    if q > 0:
        for mu in range(q):
            # Handle the cyclic pattern (eta^(q+1) = eta^1)
            mu_next = (mu + 1) % q

            # Apply f and g functions to memory patterns
            f_eta_i_next = f(eta[mu_next, :]).reshape(-1, 1)  # Column vector
            g_eta_j = g(eta[mu, :]).reshape(1, -1)  # Row vector

            # Outer product and accumulate
            W_A += np.outer(f_eta_i_next, g_eta_j)

        # Apply scaling and conditionally apply connection probability based on apply_er_to_asymmetric
        if apply_er_to_asymmetric:
            # Apply Erdős-Rényi connectivity to asymmetric component
            W_A = (c_ij / N_S) * W_A
        else:
            # Don't apply Erdős-Rényi connectivity to asymmetric component (fully connected)
            W_A = W_A / N

        # Set diagonal elements to zero
        np.fill_diagonal(W_A, 0)

    # If q=0, W_A remains zeros

    # Calculate total connectivity matrix
    W = W_S + W_A

    # Double-check that diagonal elements are zero in total matrix
    np.fill_diagonal(W, 0)

    return W_S, W_A, W, eta_raw, eta # eta_raw is the unprocessed patterns, eta is the processed patterns with φ applied

def calculate_pattern_correlation_matrix(eta_patterns):
    """
    Calculate the correlation matrix between memory patterns
    
    Parameters:
    -----------
    eta_patterns : ndarray
        Memory patterns of shape (p, N) where p is number of patterns, N is number of neurons
    
    Returns:
    --------
    correlation_matrix : ndarray
        Correlation matrix of shape (p, p)
    max_correlation : float
        Maximum absolute correlation between different patterns (excluding diagonal)
    """
    p = eta_patterns.shape[0]
    correlation_matrix = np.zeros((p, p))
    
    # Calculate all pairwise correlations
    for i in range(p):
        for j in range(p):
            if i == j:
                correlation_matrix[i, j] = 1.0  # Perfect self-correlation
            else:
                pattern_i = eta_patterns[i, :]
                pattern_j = eta_patterns[j, :]
                
                # Check for zero variance patterns
                if np.std(pattern_i) > 1e-10 and np.std(pattern_j) > 1e-10:
                    corr = np.corrcoef(pattern_i, pattern_j)[0, 1]
                    if not np.isnan(corr):
                        correlation_matrix[i, j] = corr
                    else:
                        correlation_matrix[i, j] = 0.0
                else:
                    correlation_matrix[i, j] = 0.0
    
    # Find maximum absolute correlation excluding diagonal
    off_diagonal_mask = ~np.eye(p, dtype=bool)
    if np.any(off_diagonal_mask):
        max_correlation = np.max(np.abs(correlation_matrix[off_diagonal_mask]))
    else:
        max_correlation = 0.0
    
    return correlation_matrix, max_correlation

def plot_pattern_correlation_matrix(eta_patterns, enforce_max_correlation=False, max_correlation_threshold=0.5, ax=None, output_dir='../../simulation_results'):
    """
    Plot the correlation matrix of memory patterns with constraint information
    
    Parameters:
    -----------
    eta_patterns : ndarray
        Memory patterns of shape (p, N)
    enforce_max_correlation : bool
        Whether correlation constraint was enforced
    max_correlation_threshold : float
        Maximum correlation threshold used
    ax : matplotlib.axes.Axes or None
        If provided, plot on this axis, otherwise create a new figure and axis
    output_dir : str
        Directory to save the plot image
    
    Returns:
    --------
    fig : Figure or None
        Matplotlib figure object if ax is None, otherwise None
    correlation_matrix : ndarray
        The correlation matrix
    max_correlation : float
        Maximum absolute correlation between different patterns
    """
    correlation_matrix, max_correlation = calculate_pattern_correlation_matrix(eta_patterns)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None
    
    # Plot correlation matrix with annotations
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r', 
                vmin=-1, 
                vmax=1, 
                center=0,
                ax=ax,
                cbar_kws={"label": "Correlation Coefficient"},
                square=True)
    
    # Set title with constraint information
    constraint_info = ""
    if enforce_max_correlation:
        constraint_info = f"\nConstraint: max correlation ≤ {max_correlation_threshold:.2f}"
    else:
        constraint_info = "\nNo correlation constraint applied"
    
    title = f"Memory Pattern Correlation Matrix{constraint_info}\nActual max correlation: {max_correlation:.3f}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Pattern Index')
    ax.set_ylabel('Pattern Index')
    
    # Set custom tick labels
    p = correlation_matrix.shape[0]
    ax.set_xticklabels([f'η{i+1}' for i in range(p)])
    ax.set_yticklabels([f'η{i+1}' for i in range(p)])
    plt.savefig(os.path.join(output_dir, "pattern_correlation_matrix.png"), bbox_inches='tight', dpi=300)
    plt.tight_layout()
    
    return fig, correlation_matrix, max_correlation


def plot_matrix(matrix, title, cmap='RdBu_r', ax=None):
    """
    Plot a matrix with appropriate colormap and styling
    
    Parameters:
    -----------
    matrix : ndarray
        The matrix to plot
    title : str
        Title for the plot
    cmap : str
        Colormap name
    ax : matplotlib.axes.Axes or None
        If provided, plot on this axis, otherwise create a new figure and axis
    
    Returns:
    --------
    fig : Figure or None
        Matplotlib figure object if ax is None, otherwise None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = None

    # Find the maximum absolute value for symmetric color scaling
    vmax = np.max(np.abs(matrix))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Create the heatmap
    sns.heatmap(matrix,
                cmap=cmap,
                norm=norm,
                ax=ax,
                cbar_kws={"label": "Weight Value"})

    ax.set_title(title)
    ax.set_xlabel("Neuron j")
    ax.set_ylabel("Neuron i")

    # Return the figure if created, None otherwise
    return fig