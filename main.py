import numpy as np
from experiment_runner.experiment_runner import ExperimentRunner
from contract_theory_L2.clusters import ClusterOptimization
from experiment_runner.plotter import Plotter
from fitting.q_fitting import QFitter
#  np.array([0.35, 0.2, 0.1, 0.15, 0.2])
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
    B_values = np.linspace(1000, 10000, 100)
    experiment = ExperimentRunner(cluster_optimizer, B_values)

    # experiment.run()

    # # Save the results
    # experiment.save_results("test_100.pkl")

    # Load the results (to demonstrate)
    experiment.load_results("experiment_results.pkl")

    # Example of available plots
    plotter = Plotter(experiment.results)
    # plotter.plot_results(cluster_index = 1)
    # plotter.plot_all_savings(cluster_index = 1, T = clusters_params[1]['T'])
    # plotter.plot_all_accuracies(cluster_index = 0, I = clusters_params[0]['I'], T = clusters_params[0]['T'])
    # plotter.plot_q_by_round(cluster_index = 1, I = clusters_params[1]['I'], T = clusters_params[1]['T'])
    # # # Example: Plot q vs. B for Cluster 1, User Type 1
    # plotter.plot_q_vs_B(experiment.results[1], cluster_index=1, user_type_index=0)
    # plotter.plot_q_by_type(cluster_index = 1, I = clusters_params[1]['I'], T = clusters_params[1]['T'])


    # Example how to fit a curve function on q_values and B
    # q_fitter = QFitter(experiment.results[1], cluster_optimizer.clusters[1])

    # # # Fit models and generate predictions
    # fitted_models = q_fitter.fit_q_vs_B(model="logistic")
    # B_new = np.linspace(min(q_fitter.B_values), max(q_fitter.B_values), 100)
    # # predictions = q_fitter.predict_q(fitted_models, B_new)

    # # # Plot fitted q vs. B grouped by user type
    # q_fitter.plot_fitted_q_vs_B(fitted_models, B_new)