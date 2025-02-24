import numpy as np
from experiment_runner.experiment_runner import ExperimentRunner
from contract_theory_L2.clusters import ClusterOptimization
from stackelberg_game_L1.stackelberg_solver import StackelbergSolver
from experiment_runner.plotter import Plotter
from fitting.fitting import Fitter
import os
from system_config import clusters_params
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

if __name__ == '__main__':
    
    # directory to save the results
    RESULTS_DIR = os.getenv("RESULTS_DIR")
    path_res = os.path.join(RESULTS_DIR, "C4_test_v3.pkl")


    cluster_optimizer = ClusterOptimization(clusters_params) # lower layer
    # B_values = np.linspace(0, 30000, 20)
    # Generate 40 values in the range 0-5000
    B_values_low = np.linspace(0, 1000, 25)

    # Generate 10 values in the range 5000-10000
    B_values_high = np.linspace(1000, 2000, 5)

    # Combine them
    B_values = np.concatenate((B_values_low, B_values_high))
    
    experiment = ExperimentRunner(cluster_optimizer, B_values) # overal system model

    # experiment.run()

    # # Save the results
    # experiment.save_results(path_res)

    # Load the results (to demonstrate)
    experiment.load_results(path_res)

    # Example of available plots
    # plotter = Plotter(experiment.results, clusters_params)
    # plotter.plot_results() 
    # plotter.plot_all_savings(cluster_index = 0)
    # plotter.plot_all_accuracies(cluster_index = 1)
    # plotter.plot_total_accuracy()
    # plotter.plot_q_by_round(cluster_index = 1)
    # # # Example: Plot q vs. B for Cluster 1, User Type 1
    # plotter.plot_q_vs_B(experiment.results[1], cluster_index=1, user_type_index=0)
    # plotter.plot_q_by_type(cluster_index = 0)

    # Example how to fit a curve function on Total M and B
    A_fitter = Fitter(experiment.results, experiment.cluster_optimizer)
    fitted_models = A_fitter.fit_accuracy_vs_B(model="logistic")
    B_new = np.linspace(min(A_fitter.B_values), max(A_fitter.B_values), 100)
    A_fitter.plot_fitted_total_accuracy(fitted_models, B_new)
    
    upper_layer = StackelbergSolver(fitted_models, 4)
    
    # T_values = np.linspace(0, 30000, 20)
    # T_values_low = np.linspace(0, 100, 20)

    # # Generate 10 values in the range 5000-10000
    # T_values_high = np.linspace(100, 1000, 10)

    # # Combine them
    # T_values = np.concatenate((T_values_low, T_values_high))
    
    # sol = upper_layer.solve_for_T_values(T_values)
    
    T_star = upper_layer.find_optimal_T(T_min=0, T_max=1500)





