import pickle
from contract_theory_L2.clusters import ClusterOptimization

class ExperimentRunner:
    def __init__(self, cluster_optimizer, B_values):
        self.cluster_optimizer = cluster_optimizer
        self.B_values = B_values
        self.results = []

    def run(self):
        self.results = self.cluster_optimizer.run_clusters(self.B_values)

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