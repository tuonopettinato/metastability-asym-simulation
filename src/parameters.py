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
import numpy as np

# Network size and pattern parameters
N = 1500               # Number of neurons (increased for Numba testing)
p = 4                 # Number of patterns for symmetric component
q = 4                  # Number of patterns for asymmetric component (q <= p)
c = 0.1                # Connection probability (0-1) for Erdős-Rényi model
A_S = 4.0              # Amplitude parameter for symmetric component (4.0 for p = 5, q = 3)

# f and g function (step functions) parameters for connectivity generation
f_q = 0.7              # Step value for f step function (0.7)
f_x = 2.8              # Step threshold for f step function (3.0 --- 2.8)

# using same parameters for f and g step functions in order to have a really symmetric W_S
g_q = f_q             # Step value for g step function (0.7)
g_x = f_x              # Step threshold for g step function (3.0)

# =============================================================================
# ACTIVATION FUNCTION PARAMETERS
# =============================================================================

# φ (phi) function (sigmoid) parameters - used for both connectivity and dynamics
phi_beta = 1.5                 # Steepness parameter for sigmoid function (using 1.5)
phi_r_m = 30.                  # Maximum firing rate for sigmoid function (using 30.0)
phi_x_r = 2.0                  # Threshold parameter for sigmoid function (using 2.0)

# Pattern distribution parameters
pattern_mean = 0.0      # Mean of the Gaussian distribution for memory patterns
pattern_sigma = 1.0     # Standard deviation of the Gaussian distribution for memory patterns
enforce_max_correlation = False  # Switch to enable correlation constraint enforcement
apply_phi_to_patterns = True    # Apply φ function to patterns
apply_er_to_asymmetric = False  # Apply Erdős-Rényi to asymmetric component

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Time and integration parameters
tau = 20.0              # Time constant for neural dynamics
t_start = 0.0           # Simulation start time
t_end = 2000.0           # Simulation end time # 24 000
dt = 0.2                # Time step for simulation output (increased for efficiency)

# Performance optimization
use_numba = True        # Enable Numba JIT compilation for large networks (N > 1000)
use_g = True  # Whether to apply g function to patterns, default is True in Recanatesi et al.

# Initial condition settings
init_cond_type = "Near Memory Pattern"  # Options: "Random", "Zero", "Memory Pattern", "Near Memory Pattern", "Negative Memory Pattern"
pattern_idx = 3         # Which pattern to use (0-indexed) if using pattern-based init - neglected in multiple simulations
noise_level = 0.5       # Noise level if using "Near Memory Pattern" init

# Simulation options
use_symmetric_only = True   # Whether to use only the symmetric component (W^S)
model_type = "recanatesi"   # Dynamics model: "recanatesi" or "brunel"

# Ornstein-Uhlenbeck process parameters for ζ(t)
use_ou = True          # Whether to use Ornstein-Uhlenbeck process for ζ(t)
ou_non_neg = True     # Whether to enforce non-negativity on ζ(t)
tau_zeta = 20.0          # OU time constant
zeta_bar = 0.6          # OU mean value (0.65)
sigma_zeta = 0.45        # OU noise intensity (0.65 -- 0.4 -- 0.3)
# fixed_zeta = 0.    # Constant ζ value when OU is not used, it can also be a float32 array of length equal to number of time steps
t = np.arange(t_start, t_end, dt, dtype=np.float32)
n_steps = len(t)
fixed_zeta = (zeta_bar + 4*sigma_zeta *
              np.clip(np.sin(2*np.pi*((np.arange(n_steps) % 2500)/2500) + np.pi/2) *
                      (np.arange(n_steps) // 2500 > 0),
                      0, None)
             ).astype(np.float32)

undersampling = 200





# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Number of neurons to display in plots
n_display = 10             # Maximum number of neurons to display in plots (reduced for 10k)
show_sim_plots = True  # Whether to show individual plots for each variable
plot_connectivity_matrices = True,
plot_heatmap= False,
verbose = True
ou_threshold = 2.  # Threshold for highlighting OU noise in plots
single_dir_name = "simulation_results_1"
multiple_dir_name = "multiple_simulations"

# Number of runs
runs = 1
# Import connectivity or not
import_connectivity = False
test_set = False # Wether to save in a test set folder 


# =============================================================================
# SEED
# =============================================================================
seed = 333 # Random seed for reproducibility 333

if __name__ == "__main__":
    """Create a txt file in simulation_results"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", single_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    params = {k: v for k, v in globals().items()
              if not k.startswith("__") and not callable(v)}

    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
