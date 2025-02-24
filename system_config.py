import numpy as np

# clusters_params is a list containing example configurations for multiple clusters. 
# Each cluster is represented as a dictionary with several parameters:
# - "I": Number of user types.
# - "L": Number of rounds within cluster.
# - "N": Total population or size within cluster.
# - "sigma": calibration factor.
# - "eta": calibration factor.
# - "p": A randomly generated Dirichlet distribution (representing probabilities of each user type).
# - "theta": A sorted array of random values sampled uniformly within a given range named as user types
# - "q_max": A numpy array representing the maximum quantity, computed as a sequence with equal steps.

# This configuration is used for simulation, modeling, or algorithm testing where multiple clusters with varying parameters are needed.

clusters_params = [
        {
            "I": 5, "L": 3, "N": 20, "sigma": 800, "eta": 0.9,
            "p": np.random.dirichlet(np.ones(5)), "theta": np.sort(np.random.uniform(0.2, 0.8, 5)), 
            "q_max": np.array([100 + i * 50 for i in range(3)])
        },
        {
            "I": 4, "L": 3, "N": 15, "sigma": 1000, "eta": 1,
            "p": np.random.dirichlet(np.ones(4)), "theta": np.sort(np.random.uniform(0.15, 0.7, 4)),
            "q_max": np.array([200 + i * 40 for i in range(3)])
        },
                {
            "I": 6, "L": 3, "N": 30, "sigma": 600, "eta": 0.6,
            "p": np.random.dirichlet(np.ones(6)), "theta": np.sort(np.random.uniform(0.1, 0.9, 6)), 
            "q_max": np.array([80 + i * 30 for i in range(3)])
        },
        {
            "I": 4, "L": 3, "N": 25, "sigma": 700, "eta": 0.7,
            "p": np.random.dirichlet(np.ones(4)), "theta": np.sort(np.random.uniform(0.1, 0.8, 4)),
            "q_max": np.array([200 + i * 30 for i in range(3)])
        }
    ]