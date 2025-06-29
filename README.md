# Simulating metastability through an asymmetric connectivity matrix

This project is part of my Master's Thesis in Physics of Complex Systems and aims to simulate the dynamics of a neural network based on the principles outlined by Recanatesi, Mazzucato et al. (https://arxiv.org/abs/2001.09600).

# Overview
The code is structured to allow easy configuration of parameters related to the connectivity matrices and network dynamics.
Users can modify these parameters in the `parameters.py` file to customize the simulation according to their research needs.
The simulation results are visualized using Matplotlib, and the final overlaps of memory patterns are printed to the console.
The results can also be saved as PNG files for further analysis or presentation.
To run the simulation, execute the `main.py` script. Ensure that all dependencies are installed, including NumPy and Matplotlib, as specified in the `requirements.txt` file.

# Model
The model is based on a recurrent neural network architecture that incorporates the principles of attractor dynamics. It is designed to retrieve memory patterns stored in the network's connectivity matrix. The network is initialized with a set of memory patterns, and the dynamics are governed by a set of differential equations that describe the evolution of the network's state over time. The symmetric part of the connectivity matrix is used to ensure that the network exhibits stable attractor dynamics, allowing it to converge to specific memory patterns (autoassociative retrieval), whereas the antisymmetric part introduces dynamics that can lead to heteroassociative retrieval of patterns.


