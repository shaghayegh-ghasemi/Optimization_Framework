import numpy as np
from experiment_runner.experiment_runner import ExperimentRunner
from contract_theory_L2.clusters import ClusterOptimization
from experiment_runner.plotter import Plotter

if __name__ == '__main__':
    # Example parameters for two clusters
    clusters_params = [
        {
            "I": 5, "T": 3, "N": 20, "sigma": 1000, "eta": 1,
            "p": np.random.dirichlet(np.ones(5)), "theta": np.sort(np.random.uniform(0.1, 0.9, 5)), 
            "q_max": np.array([100 + i * 50 for i in range(3)])
        },
        {
            "I": 4, "T": 3, "N": 15, "sigma": 500, "eta": 0.8,
            "p": np.random.dirichlet(np.ones(4)), "theta": np.sort(np.random.uniform(0.1, 0.7, 4)),
            "q_max": np.array([200 + i * 40 for i in range(3)])
        }
    ]

    cluster_optimizer = ClusterOptimization(clusters_params)
    B_values = np.linspace(500, 5000, 10)
    experiment = ExperimentRunner(cluster_optimizer, B_values)

    experiment.run()
    plotter = Plotter(experiment.results)
    
    plotter.plot_results(cluster_index = 1)
    plotter.plot_all_savings(cluster_index = 0, T = clusters_params[0]['T'])
    plotter.plot_all_accuracies(cluster_index = 1, I = clusters_params[1]['I'], T = clusters_params[1]['T'])
    plotter.plot_q_by_round(cluster_index = 1, I = clusters_params[1]['I'], T = clusters_params[1]['T'])
    # Example: Plot q vs. B for Cluster 1, User Type 1
    plotter.plot_q_vs_B(experiment.results[0], cluster_index=0, user_type_index=0)
    plotter.plot_q_by_type(cluster_index = 1, I = clusters_params[1]['I'], T = clusters_params[1]['T'])

