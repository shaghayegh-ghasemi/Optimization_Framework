from system_config import clusters_params, M

import numpy as np
import pickle
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()
RESULTS_DIR = os.getenv("RESULTS_DIR")

class EqualPaymentBaseline:
    def __init__(self, T_values, allocation_ratio=0.5):
        self.T_values = T_values
        self.num_servers = M
        self.allocation_ratio = allocation_ratio
        self.results_benchmark = []

    def allocate_budget(self):
        # Distribute total budget equally among local servers
        server_budget = self.total_budget / self.num_servers
        user_budget = server_budget / self.num_users_per_server

        allocation = {
            "server_budget": server_budget,
            "user_budget": user_budget
        }
        return allocation

    def run(self):
        allocation = self.allocate_budget()
        print("Equal Payment Baseline Results:")
        print(f"Each server gets: {allocation['server_budget']}")
        print(f"Each user gets: {allocation['user_budget']}")
        return allocation
    
    def save_results(self, filename):
        """
        Save the experiment results to a file using pickle.
        
        Parameters:
            filename (str): Name of the file to save the results.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.results_benchmark, f)
        print(f"Results saved to {filename}")

    def load_results(self, filename):
        """
        Load the experiment results from a file.
        
        Parameters:
            filename (str): Name of the file to load the results from.
        """
        with open(filename, 'rb') as f:
            self.results_benchmark = pickle.load(f)
        print(f"Results loaded from {filename}")
    