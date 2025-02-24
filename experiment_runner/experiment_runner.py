import pickle
from contract_theory_L2.clusters import ClusterOptimization
from fitting.fitting import Fitter
import numpy as np

class ExperimentRunner:
    def __init__(self, cluster_optimizer, B_values):
        self.cluster_optimizer = cluster_optimizer
        self.B_values = B_values
        self.results = []
        self.fitted_models = []

    def run(self):
        # run the lower layer for multiple B values
        self.results = self.cluster_optimizer.run_clusters(self.B_values)
        
        # fit a curve on Accuracy - Budget pair for each cluster
        # A_fitter = Fitter(self.results, self.cluster_optimizer)
        # self.fitted_models = A_fitter.fit_accuracy_vs_B(model="logistic")
        # B_new = np.linspace(min(A_fitter.B_values), max(A_fitter.B_values), 100)
        # A_fitter.plot_fitted_total_accuracy(self.fitted_models, B_new)
        
        # run on stackelberg and find allocated B
        # solve the optimization problem for each cluster and define optimal q for each user

    def save_results(self, filename):
        """
        Save the experiment results to a file using pickle.
        
        Parameters:
            filename (str): Name of the file to save the results.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filename}")

    def load_results(self, filename):
        """
        Load the experiment results from a file.
        
        Parameters:
            filename (str): Name of the file to load the results from.
        """
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)
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