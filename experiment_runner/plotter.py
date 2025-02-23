import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, results, clusters_params):
        self.results = results
        self.clusters_params = clusters_params

    def plot_results(self):
        """
        Plot the utility as a function of B.
        """
        
        plt.figure(figsize=(10, 7))
        
        # Iterate over all clusters
        for cluster_index, cluster_results in enumerate(self.results):
            B_values = [result[0] for result in cluster_results]
            utilities = [-1 * result[1] for result in cluster_results]
            
            plt.plot(B_values, utilities, marker='o', linestyle='-', label=f'Cluster {cluster_index + 1}')
        
        # Plot formatting
        plt.title("Local Servers' Utility vs. Budget")
        plt.xlabel(r"Budget ($\gamma_m B_m$)")
        plt.ylabel("LocalServer Utility")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_savings(self, cluster_index):
        """
        Plot savings for all rounds in one plot with different colors.

        :param cluster_results: Results for the specific cluster.
        :param opt_problem: The opt_problem object for this cluster.
        """
        L = self.clusters_params[cluster_index]['L']
        # each round in each optimization problem contains results as (B, utility, q, accuracy, savings) within each cluster
        B_values = [result[0] for result in self.results[cluster_index]]
        savings = [result[4] for result in self.results[cluster_index]]
        savings = np.array(savings)  # Shape: (len(B_values), opt_problem.T)


        plt.figure(figsize=(10, 7))
        for t in range(L):
            plt.plot(B_values, savings[:, t], marker='o', linestyle='-', label=f"Round {t+1}")
        
        plt.title(f"Savings for cluster {cluster_index + 1} vs. Budget")
        plt.xlabel(r"Budget ($\gamma_m B_m$)")
        plt.ylabel("Savings")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_accuracies(self, cluster_index):
        """
        Plot accuracy for all user types and rounds in one plot with different colors.

        :param cluster_results: Results for the specific cluster.
        :param opt_problem: The opt_problem object for this cluster.
        """
        L = self.clusters_params[cluster_index]['L']
        I = self.clusters_params[cluster_index]['I']

        B_values = [result[0] for result in self.results[cluster_index]]
        accuracy = [result[3] for result in self.results[cluster_index]]
        accuracy = np.array(accuracy)  # Shape: (len(B_values), opt_problem.I, opt_problem.T)

        plt.figure(figsize=(10, 7))
        for i in range(I):
            for t in range(L):
                plt.plot(B_values, accuracy[:, i, t], marker='o', linestyle='-', 
                         label=f"User Type {i+1}, Round {t+1}")
        
        plt.title(f"Accuracy for cluster {cluster_index + 1} vs. Budget")
        plt.xlabel(r"Budget ($\gamma_m B_m$)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_q_by_round(self, cluster_index):
        """
        Plot the trends of q values for all user types in the same plot, grouped by each round.
        """
        L = self.clusters_params[cluster_index]['L']
        I = self.clusters_params[cluster_index]['I']

        B_values = [result[0] for result in self.results[cluster_index]]
        q_values = [result[2] for result in self.results[cluster_index]]  # Extract q matrices

        for t in range(L):
            plt.figure(figsize=(10, 7))
            for i in range(I):
                q_t_i = [q[i, t] for q in q_values]
                plt.plot(B_values, q_t_i, marker='o', linestyle='-', label=rf"$q_{{{i+1}}}^{{{t+1}}}$")
            
            plt.title(rf"Trends of $q^{{{t+1}}}_i$ for All User Types in Round {t+1} - Cluster {cluster_index + 1}")
            plt.xlabel(r"Budget ($\gamma_m B_m$)")
            plt.ylabel(rf"$q^{{{t+1}}}_i$ (Contributions for Round {t+1})")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_q_vs_B(self, cluster_results, cluster_index, user_type_index):
        """
        Plot the trends of q values for specific user type in the same plot, grouped by each round.
        """
        B_values = [result[0] for result in cluster_results]
        q_values = [result[2][user_type_index, :] for result in cluster_results]
        plt.plot(B_values, q_values, marker='o')
        plt.title(f"q vs B for Cluster {cluster_index + 1}, User Type {user_type_index + 1}")
        plt.xlabel(r"Budget ($\gamma_m B_m$)")
        plt.ylabel("q Value")
        plt.grid(True)
        plt.show()

    def plot_q_by_type(self, cluster_index):
        """
        Plot the trends of q values for each user type over all rounds in the same plot.
        Each plot corresponds to a single user type and includes all rounds.
        """
        L = self.clusters_params[cluster_index]['L']
        I = self.clusters_params[cluster_index]['I']

        B_values = [result[0] for result in self.results[cluster_index]]
        q_values = [result[2] for result in self.results[cluster_index]]  # Extract q matrices

        for i in range(I):
            plt.figure(figsize=(10, 7))
            for t in range(L):
                q_t_i = [q[i, t] for q in q_values]
                plt.plot(B_values, q_t_i, marker='o', linestyle='-', label=rf"$q_{{{i+1}}}^{{{t+1}}}$")
            
            plt.title(rf"Trends of $q_{{{i+1}}}^t$ Across All Rounds - Cluster {cluster_index + 1}")
            plt.xlabel(r"Budget ($\gamma_m B_m$)")
            plt.ylabel(rf"$q_{{{i+1}}}^t$ (Contributions for User Type {i+1})")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_total_accuracy(self):
        """
        Plot the total accuracy vs. budget for a given cluster.

        :param cluster_index: Index of the cluster to compute accuracy for.
        """
        plt.figure(figsize=(10, 7))
        
        for cluster_index, cluster_results in enumerate(self.results):
            B_values = [result[0] for result in cluster_results]
            total_accuracies = [result[5] for result in cluster_results]  # Accuracy is in result[5]

            # Plot accuracy for this cluster
            plt.plot(B_values, total_accuracies, marker='o', linestyle='-', label=f"Cluster {cluster_index + 1}")
        
        plt.title(f"Total Accuracy vs. Budget")
        plt.xlabel(r"Budget ($\gamma_m B_m$)")
        plt.ylabel("Total Accuracy ($A_m$)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
