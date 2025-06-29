# Simulating metastability through an asymmetric connectivity matrix

This project is part of my Master's Thesis in Physics of Complex Systems and aims to simulate the dynamics of a neural network based on the principles outlined by Recanatesi, Mazzucato et al. (https://arxiv.org/abs/2001.09600).

# Overview
Users can modify these parameters in `parameters.py` to customize the simulation.
Results are visualized using Matplotlib, can also be saved as PNG files. The final overlaps of memory patterns are printed to the console.
To run the simulation, execute the `main.py` script.

Ensure that all dependencies are installed, including NumPy and Matplotlib, as specified in the `requirements.txt` file.

# Model
The model is based on a RNN architecture that incorporates the principles of attractor dynamics. It is designed to retrieve memory patterns stored in the network's connectivity matrix. The network is initialized with a set of memory patterns, and the dynamics are governed by a set of differential equations that describe the evolution of the network's state over time. The symmetric part of the connectivity matrix is used to ensure that the network exhibits stable attractor dynamics, allowing it to converge to specific memory patterns (autoassociative retrieval), whereas the antisymmetric part introduces dynamics that can lead to heteroassociative retrieval of patterns.

### Memory Patterns and Transfer Function

The network is initialized with a set of memory patterns $\eta_i^\mu$, where $i$ indexes the neurons and $\mu$ indexes the patterns. Values of $\eta_i^\mu$ are extracted from a Gaussian distribution with mean 0 and variance 1 without loss of generality. Anyway, mean and variance can be set in `parameters.py` with `pattern_mean` and `pattern_variance` parameters. The patterns can also be generated such that they are sparse, meaning that only a fraction of the neurons are active. This sparsity is controlled by the parameter `alpha`, which determines the proportion of active neurons in each pattern.

The activation (or transfer) function $\phi$ is then applied to these patterns to obtain the connectivity matrix components.
The activation function is a sigmoid or a ReLU, which maps currents to neural activity (firing rates).
The choice of activation function can be modified in `parameters.py` using the `activation_function` parameter.

- sigmoid
$\phi(x) = \frac{r_m}{1 + e^{-\beta(x-x_r)}}$
- ReLU 
$\phi(x) = \text{amplitude} \cdot \max(0, x)$

### Connectivity Matrix
The connectivity matrix combines symmetric and asymmetric components:

**Symmetric Component**

$$W_{ij}^S = A \cdot \left(\frac{c_{ij}}{cN}\right) \cdot \sum_{\mu=1}^{p} f(\phi(\eta_i^\mu)) g(\phi(\eta_j^\mu))$$

Where:
- $N$: Number of neurons
- $A$: Scaling factor for the symmetric component
- $c_{ij}$: Connection probability between neurons $i$ and $j$
- $cN$: Normalization factor (total connections)
- $p$: Number of memory patterns for the symmetric component
- $f$ and $g$: Post-synaptic and pre-synaptic functions
- Erdős-Rényi random graph is used to generate the connectivity matrix: connections are made with probability $c < 1$


**Asymmetric Component**


$$W_{ij}^A = \left(\frac{1}{N}\right) \cdot \sum_{\mu=1}^{q} f(\phi(\eta_i^{\mu+1})) g(\phi(\eta_j^\mu))$$

Where:
- $q$: Number of patterns for the asymmetric component ($q \leq p$)
- $\mu+1$: Indexing for the asymmetric component, ensuring that the patterns are shifted by one index
- Erdős-Rényi is not used unless specified in the parameters file (`apply_er_to_asymmetric`)

### Network Dynamics

The dynamics of the network is described by the following differential equation for each neuron current $u_i(t)$:

$$\frac{du_i(t)}{dt} = -\frac{u_i(t)}{\tau} + \sum_{j=1}^{N} W_{ij}^S \phi(u_j(t)) + \zeta(t)\cdot \sum_{j=1}^{N} W_{ij}^A \phi(u_j(t))$$

Where:
- $u_i(t)$: Current of neuron $i$ at time $t$
- $\tau$: Time constant for the current decay
- $\zeta(t)$: Ornstein-Uhlenbeck control signal
