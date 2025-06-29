#!/usr/bin/env python3
"""
Configuration file for neural network simulations
This file contains all parameters for connectivity matrices and network dynamics.
Edit this file to configure your simulation, then run main.py to execute.
"""

# =============================================================================
# CONNECTIVITY MATRIX PARAMETERS
# =============================================================================

# Network size and pattern parameters
N = 1000               # Number of neurons (increased for Numba testing)
p = 5                 # Number of patterns for symmetric component
q = 3                  # Number of patterns for asymmetric component (q <= p)
c = 0.1                # Connection probability (0-1) for Erdős-Rényi model
A_S = 3.0              # Amplitude parameter for symmetric component

# φ (phi) function parameters - used for both connectivity and dynamics
phi_function_type = "sigmoid"  # "sigmoid" or "relu"
phi_amplitude = 1.0            # Amplitude parameter for ReLU function
phi_beta = 1.5                 # Steepness parameter for sigmoid function
phi_r_m = 30.0                  # Maximum firing rate for sigmoid function
phi_x_r = 2.0                  # Threshold parameter for sigmoid function

# f and g function parameters for connectivity generation
f_type = "step"      # Type of f function ("sigmoid" or "step")
f_r_m = 1.0            # Maximum firing rate for f sigmoid
f_beta = 1.0           # Steepness for f sigmoid
f_x_r = 0.0            # Threshold for f sigmoid
f_q = 0.7              # Step value for f step function
f_x = 3.0              # Step threshold for f step function

g_type = "step"      # Type of g function ("sigmoid" or "step") 
g_r_m = 1.0            # Maximum firing rate for g sigmoid
g_beta = 1.0           # Steepness for g sigmoid
g_x_r = 0.0            # Threshold for g sigmoid
g_q = 0.7              # Step value for g step function
g_x = 3.0              # Step threshold for g step function

# Pattern distribution parameters
pattern_mean = 0.0      # Mean of the Gaussian distribution for memory patterns
pattern_sigma = 1.0     # Standard deviation of the Gaussian distribution for memory patterns
alpha = 1.0  # Memory pattern sparsity (0-1): fraction of active neurons per pattern
enforce_max_correlation = False  # Switch to enable correlation constraint enforcement
max_correlation = 0.5  # Maximum allowed correlation between patterns (when enforcement enabled)
apply_sigma_cutoff = False       # Apply 1σ cutoff to patterns
apply_phi_to_patterns = True    # Apply φ function to patterns
apply_er_to_asymmetric = False  # Apply Erdős-Rényi to asymmetric component

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Time and integration parameters
tau = 20.0              # Time constant for neural dynamics
t_start = 0.0           # Simulation start time
t_end = 1000.0           # Simulation end time
dt = 0.2                # Time step for simulation output (increased for efficiency)

# Performance optimization
use_numba = False        # Enable Numba JIT compilation for large networks (N > 1000)
use_g = True  # Whether to apply g function to patterns, default is True in Recanatesi et al. 

# Initial condition settings
init_cond_type = "Near Memory Pattern"  # Options: "Random", "Zero", "Memory Pattern", "Near Memory Pattern"
pattern_idx = 0         # Which pattern to use (0-indexed) if using pattern-based init
noise_level = 0.5       # Noise level if using "Near Memory Pattern" init

# Simulation options
use_symmetric_only = False   # Whether to use only the symmetric component (W^S)
model_type = "recanatesi"   # Dynamics model: "recanatesi" or "brunel"

# Ornstein-Uhlenbeck process parameters for ζ(t)
use_ou = True          # Whether to use Ornstein-Uhlenbeck process for ζ(t)
tau_zeta = 20.0          # OU time constant
zeta_bar = 0.6          # OU mean value
sigma_zeta = 0.3        # OU noise intensity
constant_zeta = 0.5     # Constant ζ value when OU is not used

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Number of neurons to display in plots
n_display = 10             # Maximum number of neurons to display in plots (reduced for 10k)


# =============================================================================
# SEED
# =============================================================================
seed = 20 # Random seed for reproducibility
