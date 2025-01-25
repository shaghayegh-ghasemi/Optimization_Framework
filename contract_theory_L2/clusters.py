import numpy as np
from contract_theory_L2.optimization_model import OptimizationContractTheory

class ClusterOptimization:
    def __init__(self, clusters_params):
        """
        Initialize ClusterOptimization with multiple clusters  (Handles multiple clusters).
        - clusters_params: List of dictionaries, where each dict contains the parameters for a cluster.
        """
        self.clusters = []
        for params in clusters_params:
            cluster = OptimizationContractTheory(
                I=params["I"],
                T=params["T"],
                N=params["N"],
                sigma=params["sigma"],
                eta=params["eta"],
                p=params["p"],
                theta=params["theta"],
                q_max=params["q_max"]
            )
            self.clusters.append(cluster)

    def run_clusters(self, B_values):
        cluster_results = []
        for cluster in self.clusters:
            cluster_results.append(self.run_experiment_for_cluster(cluster, B_values))
        return cluster_results

    def run_experiment_for_cluster(self, cluster, B_values):
        results = []
        for B in B_values:
            q, utility, accuracy, savings, total_accuracy = cluster.solve(B)
            results.append((B, utility, q, accuracy, savings, total_accuracy))
        return results

    def parametric_relation_q_B(self, cluster_index, B_values):
        cluster = self.clusters[cluster_index]
        q_B_relations = []
        for B in B_values:
            q, _, _, _ = cluster.solve(B)
            q_B_relations.append(q)
        return np.array(q_B_relations)