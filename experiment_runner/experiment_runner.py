import numpy as np
from contract_theory_L2.clusters import ClusterOptimization

class ExperimentRunner:
    def __init__(self, cluster_optimizer, B_values):
        self.cluster_optimizer = cluster_optimizer
        self.B_values = B_values
        self.results = []

    def run(self):
        self.results = self.cluster_optimizer.run_clusters(self.B_values)