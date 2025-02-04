import numpy as np
from experiment_runner.experiment_runner import ExperimentRunner
from contract_theory_L2.clusters import ClusterOptimization
from experiment_runner.plotter import Plotter
from fitting.fitting import Fitter
import os

if __name__ == '__main__':
    RESULTS_DIR = os.getenv("RESULTS_DIR")

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
    B_values = np.linspace(1000, 10000, 50)
    experiment = ExperimentRunner(cluster_optimizer, B_values)

    # experiment.run()

    # # Save the results
    # experiment.save_results("test_50.pkl")

    # Load the results (to demonstrate)
    experiment.load_results("test_50.pkl")

    # Example of available plots
    plotter = Plotter(experiment.results, clusters_params)
    # plotter.plot_results(cluster_index = 1)
    # plotter.plot_all_savings(cluster_index = 1)
    # plotter.plot_all_accuracies(cluster_index = 1)
    plotter.plot_total_accuracy(cluster_index = 0)
    # plotter.plot_q_by_round(cluster_index = 1)
    # # # Example: Plot q vs. B for Cluster 1, User Type 1
    # plotter.plot_q_vs_B(experiment.results[1], cluster_index=1, user_type_index=0)
    # plotter.plot_q_by_type(cluster_index = 1)


    # Example how to fit a curve function on q_values and B
    # q_fitter = Fitter(experiment.results[1], cluster_optimizer.clusters[1])
    # fitted_models = q_fitter.fit_q_vs_B(model="logistic")
    # B_new = np.linspace(min(q_fitter.B_values), max(q_fitter.B_values), 100)
    # # predictions = q_fitter.predict_q(fitted_models, B_new)
    # q_fitter.plot_fitted_q_vs_B(fitted_models, B_new)

    # Example how to fit a curve function on Total M and B
    A_fitter = Fitter(experiment.results[0], cluster_optimizer.clusters[0])

    fitted_models = A_fitter.fit_accuracy_vs_B(model="logistic")

    B_new = np.linspace(min(A_fitter.B_values), max(A_fitter.B_values), 100)

    A_fitter.plot_fitted_total_accuracy(fitted_models, B_new)




