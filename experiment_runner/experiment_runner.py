from system_config import clusters_params, M
from contract_theory_L2.clusters import ClusterOptimization
from stackelberg_game_L1.stackelberg_solver import StackelbergSolver
from fitting.fitting import Fitter

import pickle
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()
RESULTS_DIR = os.getenv("RESULTS_DIR")

class ExperimentRunner:
    def __init__(self):
        self.results_L2 = []
        self.fitted_models = []
        self.optimal_results_L2 = []
        self.optimal_results_L1 = []

    def run(self, B_values, iteration = 5, path = None):
        print("Running ... !")
        lower_layer = ClusterOptimization(clusters_params) # lower layer
        
        if path == None:
            # run the lower layer for multiple B values and fit a curve on total accuracy and B value pairs for each cluster
            self.results_L2 = lower_layer.run_clusters(B_values)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            path_res = os.path.join(RESULTS_DIR, f"{timestamp}.pkl")
            self.save_results(path_res)
            print("Completed the results for differebt B_values and saved them.")
        else:
            self.load_results(path)
            print("Loaded the results.")
        
        # fit a curve on Accuracy - Budget pair for each cluster
        A_fitter = Fitter(self.results_L2)
        self.fitted_models = A_fitter.fit_accuracy_vs_B(model="logistic")
        B_new = np.linspace(min(A_fitter.B_values), max(A_fitter.B_values), 100)
        A_fitter.plot_fitted_total_accuracy(self.fitted_models, B_new)
        
        print("Fitted the Accuracy Curve.")
        
        upper_layer = StackelbergSolver(self.fitted_models, M)
        
        for i in range(iteration):
            print(f"Started the ietarion {i}.")
            T_star = upper_layer.find_optimal_T(T_min=0, T_max=8500)
            res_star = upper_layer.solve_system(T_star)
            
            gamma_values = res_star[:4]
            accuracy_values = res_star[4:]
            total_A = max(np.sum(accuracy_values), T_star / 10)  # Ensure a minimum meaningful scale
            allocated_B = np.maximum(gamma_values * (accuracy_values / total_A) * T_star * np.linspace(0.8, 1.2, M), 1e-3)
            
            # save the optimal values from upper layer for this iteration
            self.optimal_results_L1.append((allocated_B, T_star, gamma_values, accuracy_values))
            print("Found the optimal value for Stackelberg - upper layer.")
            
            # solve the optimization problem for each cluster and define optimal q for each user
            self.optimal_results_L2.append(lower_layer.solve_experiment_B(allocated_B))
            print("Found the optimal value for Contract Theory - lower layer.")
            
            
        print(f'optimal results for upper layer: {self.optimal_results_L1}')
        print(f'optimal results for lower layer: {self.optimal_results_L2}')

    def save_results(self, filename):
        """
        Save the experiment results to a file using pickle.
        
        Parameters:
            filename (str): Name of the file to save the results.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.results_L2, f)
        print(f"Results saved to {filename}")

    def load_results(self, filename):
        """
        Load the experiment results from a file.
        
        Parameters:
            filename (str): Name of the file to load the results from.
        """
        with open(filename, 'rb') as f:
            self.results_L2 = pickle.load(f)
        print(f"Results loaded from {filename}")
        
    # def calculate_fitted_models(self, results, cluster_optimizer, model="logistic"):
    #     """
    #     Calculate fitted model functions for each cluster and store them in a list.

    #     Parameters:
    #         results: The list containing results.
    #         cluster_optimizer (ClusterOptimization): The cluster optimizer containing clusters.
    #         model (str): The model to fit. Options: "logistic", "exp_decay".

    #     Returns:
    #         list: List of fitted model functions [fitted_model_0, fitted_model_1, ..., fitted_model_M].
    #     """
    #     fitted_models = []

    #     for i, cluster in enumerate(cluster_optimizer.clusters):
    #         print(f"Fitting {model} model for accuracy vs. budget for cluster {i}...")

    #         # Create a Fitter object for the current cluster
    #         A_fitter = Fitter(results[i], cluster)
            
    #         # Get the fitted model function for the specified model
    #         fitted_model = A_fitter.fit_accuracy_vs_B(model=model)
            
    #         # Plot each fitting
    #         # B_new = np.linspace(min(A_fitter.B_values), max(A_fitter.B_values), 100)
    #         # A_fitter.plot_fitted_total_accuracy(fitted_model, B_new)

    #         # Append the fitted model function to the list
    #         fitted_models.append(fitted_model)

    #     return fitted_models