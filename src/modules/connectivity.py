"""
connectivity.py contains functions to generate and analyze connectivity matrices
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from modules.activation import threshold_function, relu_function, sigmoid_function, step_function

def generate_connectivity_matrix(N,
                                 p,
                                 q,
                                 c,
                                 A_S,
                                 f_q=1.0,
                                 f_x=0.0,
                                 g_q=1.0,
                                 g_x=0.0,
                                 pattern_mean=0.0,
                                 pattern_sigma=1.0,
                                 phi_beta=1.0,
                                 phi_r_m=1.0,
                                 phi_x_r=0.0,
                                 apply_phi_to_patterns=True,
                                 apply_er_to_asymmetric=False,
                                 enforce_max_correlation=False):
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
    f_q : float
        Step value for f step function
    f_x : float
        Step threshold for f step function
    g_q : float
        Step value for g step function
    g_x : float
        Step threshold for g step function
    phi_beta : float
        Steepness parameter for sigmoid φ function
    phi_r_m : float
        Maximum firing rate for sigmoid φ function  
    phi_x_r : float
        Threshold parameter for sigmoid φ function
    pattern_mean : float
        Mean of the Gaussian distribution for memory patterns
    pattern_sigma : float
        Standard deviation of the Gaussian distribution for memory patterns
    apply_phi_to_patterns : bool
        Whether to apply the activation function (φ) to the patterns, default: True
    apply_er_to_asymmetric : bool
        Whether to apply Erdős-Rényi connectivity to asymmetric component. 
        If True: Apply c_ij to both symmetric and asymmetric components.
        If False: Apply c_ij only to symmetric component, default: False
    enforce_max_correlation : bool
        Whether to enforce correlation constraint by selecting patterns with low mutual correlation
        from a larger pool, default: False
    
    Returns:
    --------
    W_S : ndarray
        Symmetric component of connectivity matrix
    W_A : ndarray
        Asymmetric component of connectivity matrix
    W : ndarray
        Total connectivity matrix (W_S + W_A)
    eta_raw : ndarray
        Raw memory patterns before applying φ function
    eta : ndarray
        Memory patterns used to generate the connectivity matrices
    """
    # Set up f and g functions, which are both step functions
    f = lambda x: step_function(x, f_q, f_x)
    g = lambda x: step_function(x, g_q, g_x)
    # Set up φ function, which is a sigmoid
    phi = lambda x: sigmoid_function(x, phi_r_m, phi_beta, phi_x_r)

    # Generate random memory patterns from Gaussian distribution
    
    # =================== Memory patterns ======================
    if enforce_max_correlation and p > 1:
        N_internal = N
        pool_size = 1000

        # Genera pool di pattern raw
        pool = np.random.normal(pattern_mean, pattern_sigma, size=(pool_size, N_internal))
        if apply_phi_to_patterns:
            phi_pool = np.zeros_like(pool)
            for mu in range(pool_size):
                phi_pool[mu, :] = phi(pool[mu, :])
        else:
            phi_pool = pool.copy()

        # Calcola g(phi(pool)) per selezione
        g_phi_pool = np.zeros_like(pool)
        for mu in range(pool_size):
            g_phi_pool[mu, :] = g(phi_pool[mu, :])

        # Seleziona p pattern meno correlati in g_phi_pool
        selected_indices = []
        remaining_indices = list(range(pool_size))
        while len(selected_indices) < p and remaining_indices:
            if not selected_indices:
                idx = np.random.choice(remaining_indices)
                selected_indices.append(idx)
                remaining_indices.remove(idx)
            else:
                correlations = []
                for idx in remaining_indices:
                    candidate = g_phi_pool[idx, :]
                    corr_avg = np.mean([abs(np.corrcoef(candidate, g_phi_pool[i, :])[0, 1]) for i in selected_indices])
                    correlations.append(corr_avg)
                idx_min = remaining_indices[np.argmin(correlations)]
                selected_indices.append(idx_min)
                remaining_indices.remove(idx_min)

        eta_raw = pool[selected_indices, :]
        if apply_phi_to_patterns:
            eta = np.zeros_like(eta_raw)
            for mu in range(p):
                eta[mu, :] = phi(eta_raw[mu, :])
        else:
            eta = eta_raw.copy()
    else:
        # Caso standard: generazione diretta
        eta_raw = np.random.normal(pattern_mean, pattern_sigma, size=(p, N))
        if apply_phi_to_patterns:
            eta = np.zeros_like(eta_raw)
            for mu in range(p):
                eta[mu, :] = phi(eta_raw[mu, :])
        else:
            eta = eta_raw.copy()

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

    # Initialize symmetric component
    W_S = np.zeros((N, N))

    # Calculate symmetric component
    for mu in range(p):
        # Apply f and g functions to memory patterns
        f_eta_i = f(eta[mu, :]).reshape(-1, 1)  # Column vector
        g_eta_j = g(eta[mu, :]).reshape(1, -1)  # Row vector

        # centering 
        # f_eta_i -= np.mean(f_eta_i)
        # f_eta_i = f_eta_i/(np.std(f_eta_i) + 1e-12)
        # g_eta_j -= np.mean(g_eta_j)
        # g_eta_j = g_eta_j/(np.std(g_eta_j) + 1e-12)

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

def calculate_pattern_correlation_matrix(patterns):
    """
    Calculate the correlation matrix between memory patterns
    
    Parameters:
    -----------
    patterns : ndarray
        Memory patterns of shape (p, N) where p is number of patterns, N is number of neurons
    
    Returns:
    --------
    correlation_matrix : ndarray
        Correlation matrix of shape (p, p)
    max_correlation : float
        Maximum absolute correlation between different patterns (excluding diagonal)
    """
    p = patterns.shape[0]
    correlation_matrix = np.zeros((p, p))
    
    # Calculate all pairwise correlations
    for i in range(p):
        for j in range(p):
            if i == j:
                correlation_matrix[i, j] = 1.0  # Perfect self-correlation
            else:
                pattern_i = patterns[i, :]
                pattern_j = patterns[j, :]
                
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

def plot_pattern_correlation_matrix(patterns, enforce_max_correlation=False, ax=None, output_dir='../../simulation_results'):
    """
    Plot the correlation matrix of memory patterns with constraint information
    
    Parameters:
    -----------
    patterns : ndarray
        Memory patterns of shape (p, N)
    enforce_max_correlation : bool
        Whether correlation constraint was enforced
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
    correlation_matrix, max_correlation = calculate_pattern_correlation_matrix(patterns)
    # set NaN on the diagonal for better visualization
    np.fill_diagonal(correlation_matrix, np.nan)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None
    
    # Plot correlation matrix with annotations
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.4f',
                cmap='RdBu_r', 
                vmin=-max_correlation, 
                vmax=max_correlation, 
                center=0,
                ax=ax,
                #cbar_kws={"label": "Correlation Coefficient"},
                square=True)
    
    # Set title with constraint information
    constraint_info = ""
    if enforce_max_correlation:
        constraint_info = f"\nConstraint: pooling correlation enforced"
    else:
        constraint_info = "\nNo correlation constraint applied"
    
    #title = f"Memory Pattern Correlation Matrix{constraint_info}\nActual max correlation: {max_correlation:.3f}"
    # ax.set_xlabel('Pattern Index', fontsize=18)
    # ax.set_ylabel('Pattern Index', fontsize=18)
    
    # Set custom tick labels
    p = correlation_matrix.shape[0]
    ax.set_xticklabels([f'η{i+1}' for i in range(p)], fontsize=14)
    ax.set_yticklabels([f'η{i+1}' for i in range(p)], fontsize=14)
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

    # Replace NaN or inf with zeros to avoid plotting issues
    if not np.isfinite(matrix).all():
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute symmetric normalization range
    vmax = float(np.max(np.abs(matrix)))
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1e-9  # Prevent TwoSlopeNorm from crashing

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Create the heatmap
    sns.heatmap(
        matrix,
        cmap=cmap,
        norm=norm,
        ax=ax,
        cbar_kws={"label": "Weight Value"},
        square=True,
        xticklabels=False,
        yticklabels=False
    )

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Neuron j", fontsize=18)
    ax.set_ylabel("Neuron i", fontsize=18)

    # Return the figure if created, None otherwise
    return fig