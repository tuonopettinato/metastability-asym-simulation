"""
This file intends to simulate the dynamics of the NN with many different initial conditions
"""

import os



from main import simulation

def multiple_simulations():
    os.makedirs("multiple_simulations", exist_ok=True)
    for seed in range(5):  # Run 5 different simulations
        # transform the number of the seed into a string called addition
        addition = str(seed)
        simulation(seed=seed, addition=addition)

if __name__ == "__main__":
    multiple_simulations()