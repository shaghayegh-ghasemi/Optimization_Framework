import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, results):
        self.results = results

    def plot_results(self, cluster_index):
        """
        Plot the utility as a function of B.
        """
        B_values = [result[0] for result in self.results[cluster_index]]
        utilities = [-1*result[1] for result in self.results[cluster_index]]

        plt.figure(figsize=(8, 6))
        plt.plot(B_values, utilities, marker='o', linestyle='-', color='b')
        plt.title("Utility vs. Budget (B)")
        plt.xlabel("Budget (B)")
        plt.ylabel("Server Utility")
        plt.grid(True)
        plt.show()

    def plot_all_savings(self, cluster_index, T):
        """
        Plot savings for all rounds in one plot with different colors.

        :param cluster_results: Results for the specific cluster.
        :param opt_problem: The opt_problem object for this cluster.
        """

        # each round in each optimization problem contains results as (B, utility, q, accuracy, savings) within each cluster
        B_values = [result[0] for result in self.results[cluster_index]]
        savings = [result[4] for result in self.results[cluster_index]]
        savings = np.array(savings)  # Shape: (len(B_values), opt_problem.T)


        plt.figure(figsize=(10, 7))
        for t in range(T):
            plt.plot(B_values, savings[:, t], marker='o', linestyle='-', label=f"Round {t+1}")
        
        plt.title("Savings vs. Budget (B)")
        plt.xlabel("Budget (B)")
        plt.ylabel("Savings")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_accuracies(self, cluster_index, I, T):
        """
        Plot accuracy for all user types and rounds in one plot with different colors.

        :param cluster_results: Results for the specific cluster.
        :param opt_problem: The opt_problem object for this cluster.
        """
        
        B_values = [result[0] for result in self.results[cluster_index]]
        accuracy = [result[3] for result in self.results[cluster_index]]
        accuracy = np.array(accuracy)  # Shape: (len(B_values), opt_problem.I, opt_problem.T)

        plt.figure(figsize=(10, 7))
        for i in range(I):
            for t in range(T):
                plt.plot(B_values, accuracy[:, i, t], marker='o', linestyle='-', 
                         label=f"User Type {i+1}, Round {t+1}")
        
        plt.title("Accuracy vs. Budget (B)")
        plt.xlabel("Budget (B)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_q_by_round(self, cluster_index, I, T):
        """
        Plot the trends of q values for all user types in the same plot, grouped by each round.
        """
        B_values = [result[0] for result in self.results[cluster_index]]
        q_values = [result[2] for result in self.results[cluster_index]]  # Extract q matrices

        for t in range(T):
            plt.figure(figsize=(10, 7))
            for i in range(I):
                q_t_i = [q[i, t] for q in q_values]
                plt.plot(B_values, q_t_i, marker='o', linestyle='-', label=rf"$q_{{{i+1}}}^{{{t+1}}}$")
            
            plt.title(rf"Trends of $q^{{{t+1}}}_i$ for All User Types in Round {t+1}")
            plt.xlabel("Budget (B)")
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
        plt.xlabel("Budget (B)")
        plt.ylabel("q Value")
        plt.grid(True)
        plt.show()

    def plot_q_by_type(self, cluster_index, I, T):
        """
        Plot the trends of q values for each user type over all rounds in the same plot.
        Each plot corresponds to a single user type and includes all rounds.
        """

        B_values = [result[0] for result in self.results[cluster_index]]
        q_values = [result[2] for result in self.results[cluster_index]]  # Extract q matrices

        for i in range(I):
            plt.figure(figsize=(10, 7))
            for t in range(T):
                q_t_i = [q[i, t] for q in q_values]
                plt.plot(B_values, q_t_i, marker='o', linestyle='-', label=rf"$q_{{{i+1}}}^{{{t+1}}}$")
            
            plt.title(rf"Trends of $q_{{{i+1}}}^t$ Across All Rounds")
            plt.xlabel("Budget (B)")
            plt.ylabel(rf"$q_{{{i+1}}}^t$ (Contributions for User Type {i+1})")
            plt.legend()
            plt.grid(True)
            plt.show()