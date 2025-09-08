#!/usr/bin/env python3
"""
Configuration file for neural network simulations
This file contains all parameters for connectivity matrices and network dynamics.
Edit this file to configure your simulation, then run main.py to execute.
"""

# =============================================================================
# CONNECTIVITY MATRIX PARAMETERS
# =============================================================================

import os 
from scipy.__config__ import show

# Network size and pattern parameters
N = 1300               # Number of neurons (increased for Numba testing)
p = 4                 # Number of patterns for symmetric component
q = 4                  # Number of patterns for asymmetric component (q <= p)
c = 0.1                # Connection probability (0-1) for Erdős-Rényi model
A_S = 4.0              # Amplitude parameter for symmetric component (4.0 for p = 5, q = 3)

# φ (phi) function parameters - used for both connectivity and dynamics
phi_function_type = "sigmoid"  # "sigmoid" or "relu"
phi_amplitude = 1.0            # Amplitude parameter for ReLU function
phi_beta = 1.5                 # Steepness parameter for sigmoid function
phi_r_m = 30.0                  # Maximum firing rate for sigmoid function (using 30.0)
phi_x_r = 2.0                  # Threshold parameter for sigmoid function

# f and g function parameters for connectivity generation
f_type = "step"      # Type of f function ("sigmoid" or "step") (step)
f_r_m = 1.0            # Maximum firing rate for f sigmoid
f_beta = 1.0           # Steepness for f sigmoid
f_x_r = 0.0            # Threshold for f sigmoid
f_q = .7              # Step value for f step function (0.7)
f_x = 2.8              # Step threshold for f step function (3.0)

g_type = "step"      # Type of g function ("sigmoid" or "step") (step)
g_r_m = 1.0            # Maximum firing rate for g sigmoid
g_beta = 1.0           # Steepness for g sigmoid
g_x_r = 0.0            # Threshold for g sigmoid
g_q = .7              # Step value for g step function (0.7)
g_x = 2.8              # Step threshold for g step function (3.0)

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
t_end = 30000.0           # Simulation end time
dt = 0.4                # Time step for simulation output (increased for efficiency)

# Performance optimization
use_numba = True        # Enable Numba JIT compilation for large networks (N > 1000)
use_g = True  # Whether to apply g function to patterns, default is True in Recanatesi et al.

# Initial condition settings
init_cond_type = "Near Memory Pattern"  # Options: "Random", "Zero", "Memory Pattern", "Near Memory Pattern"
pattern_idx = 1         # Which pattern to use (0-indexed) if using pattern-based init
noise_level = 0.5       # Noise level if using "Near Memory Pattern" init

# Simulation options
use_symmetric_only = False   # Whether to use only the symmetric component (W^S)
model_type = "recanatesi"   # Dynamics model: "recanatesi" or "brunel"

# Ornstein-Uhlenbeck process parameters for ζ(t)
use_ou = True          # Whether to use Ornstein-Uhlenbeck process for ζ(t)
tau_zeta = 20.0          # OU time constant
zeta_bar = 0.65          # OU mean value (0.6 for p = 5, q = 3)
sigma_zeta = 0.65        # OU noise intensity (0.3 for p = 5, q = 3)
constant_zeta = 1.     # Constant ζ value when OU is not used

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Number of neurons to display in plots
n_display = 10             # Maximum number of neurons to display in plots (reduced for 10k)
show_sim_plots = False  # Whether to show individual plots for each variable
plot_connectivity_matrices = False,
plot_heatmap= False,
verbose = False
single_dir_name = "simulation_results"
multiple_dir_name = "multiple_simulations"

# Number of runs
runs = 8

# =============================================================================
# SEED
# =============================================================================
seed = 3 # Random seed for reproducibility

if __name__ == "__main__":
    """Create a txt file in simulation_results"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "simulation_results")
    os.makedirs(output_dir, exist_ok=True)

    params = {k: v for k, v in globals().items()
              if not k.startswith("__") and not callable(v)}

    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
