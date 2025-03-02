import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # Stable implementation of sigmoid
from scipy.stats import linregress
import matplotlib.pyplot as plt

class Fitter:
    def __init__(self, results):
        """
        Initialize QFitter with results and optimization problem.

        Parameters:
            results (list): Results containing B-values, q-values, and other outputs.
            opt_problem (object): Optimization problem object defining I (user types) and T (rounds).
        """
        self.results = results
        # self.opt_problem = opt_problem
        self.B_values = [result[0] for result in results[0]]  # Extract B-values
        # self.q_values = [result[2] for result in results]  # Extract q matrices
        # self.total_accuracies = [result[5] for result in results]  # Extract total accuracies

    @staticmethod
    def logistic(B, L, k, B0, v):
        """Improved logistic function with extra flexibility to fit all data points."""
        B = np.maximum(B, 1e-6)  # Prevent division instability at B=0
        return L * (B / (B + B0)) * (1 / (1 + np.exp(-k * (B - B0))) ** v)

    @staticmethod
    def exp_decay(B, L, k):
        """Exponential growth with asymptote."""
        # Clip B to prevent overflow
        B = np.clip(B, 0, 1e3)
        return L * (1 - np.exp(-k * B))
    
    @staticmethod
    def generalized_logistic(B, A_max, k, B_0, v):
        """Generalized logistic function that ensures A_m(0) = 0."""
        B = np.maximum(B, 1e-6)  # Prevent division instability at B=0
        return A_max * (B / (B + B_0)) * (1 / (1 + np.exp(-k * (B - B_0))) ** v)

    def piecewise_linear_logistic(self, B, A_linear_slope, A_linear_intercept, A_max, k, B_0):
        """ 
        Piecewise function: Linear for small B, Logistic for large B.
        B_critical is chosen dynamically.
        """
        B_critical = np.median(B)  # Choose a threshold dynamically
        A_logistic = A_max / (1 + np.exp(-k * (B - B_0)))
        
        # Apply piecewise behavior
        A_piecewise = np.where(B < B_critical, A_linear_slope * B + A_linear_intercept, A_logistic)
        return A_piecewise
    
    def fit_accuracy_vs_B(self, model="logistic"):
        """
        Fit the summation of accuracies (A_m) as a function of B for all clusters.

        Parameters:
            model (str): The type of model to fit ('logistic', 'exp_decay', 'generalized_logistic', 'piecewise').

        Returns:
            list: A list of fitted model functions, one per cluster.
        """
        fitted_models = []  # Store models for each cluster

        for cluster_results in self.results:
            B_values = np.array([result[0] for result in cluster_results])  # Extract B-values
            A_m_values = np.array([result[5] for result in cluster_results])  # Extract total accuracies

            if model == "logistic":
                params, _ = curve_fit(
                    self.logistic, B_values, A_m_values,
                    p0=[max(A_m_values), 0.01, np.median(B_values), 1],  # Adding v parameter
                    bounds=([0, 1e-6, 0, 0.1], [max(A_m_values) * 2, 1, max(B_values), 5])  # More robust bounds
                )
                fitted_model = lambda B_new, p=params: self.logistic(B_new, *p)

            elif model == "exp_decay":
                params, _ = curve_fit(
                    self.exp_decay, B_values, A_m_values,
                    p0=[max(A_m_values), 1e-4], maxfev=10000
                )
                fitted_model = lambda B_new, p=params: self.exp_decay(B_new, *p)

            elif model == "generalized_logistic":
                params, _ = curve_fit(
                    self.generalized_logistic, B_values, A_m_values,
                    p0=[np.max(A_m_values), 0.001, np.min(B_values), 0.5],  # More stable initial values
                    bounds=([0, 1e-6, 0, 0.1], [np.max(A_m_values) * 2, 1, np.max(B_values), 5]),
                    maxfev=10000
                )
                fitted_model = lambda B_new, p=params: self.generalized_logistic(B_new, *p)

            elif model == "piecewise":
                # Fit a linear model for the small-budget region
                B_critical = np.median(B_values)
                small_B_mask = B_values < B_critical
                slope, intercept, _, _, _ = linregress(B_values[small_B_mask], A_m_values[small_B_mask])

                # Fit the logistic model for large-budget region
                params, _ = curve_fit(
                    self.logistic, B_values, A_m_values,
                    p0=[max(A_m_values), 0.01, np.median(B_values)]
                )

                # Combine models
                fitted_model = lambda B_new, p=params: self.piecewise_linear_logistic(B_new, slope, intercept, *p)

            else:
                raise ValueError(f"Unsupported model type: {model}")

            fitted_models.append(fitted_model)  # Store the model for this cluster

        return fitted_models  # Return a list of models, one per cluster

    def plot_fitted_total_accuracy(self, fitted_models, B_new):
        """
        Plot the fitted total accuracy (A_m) vs. Budget (B) for all clusters.

        Parameters:
            fitted_models (list): A list of fitted models for each cluster.
            B_new (numpy array): Range of budget values for prediction.
        """
        plt.figure(figsize=(10, 7))

        for cluster_index, (cluster_results, fitted_model) in enumerate(zip(self.results, fitted_models)):
            B_values = np.array([result[0] for result in cluster_results])  # Budget values
            A_m_values = np.array([result[5] for result in cluster_results])  # Total accuracies
            A_m_pred = fitted_model(B_new)  # Predict accuracy using the fitted model

            # Plot observed values
            plt.scatter(B_values, A_m_values, label=f"Observed $A_m$ - Cluster {cluster_index}")

            # Plot fitted curve
            plt.plot(B_new, A_m_pred, linestyle='-', label=f"Fitted Curve - Cluster {cluster_index}")

        # Formatting
        plt.title("Summation of Accuracies ($A_m$) vs. Budget for All Clusters")
        plt.xlabel("Budget (B)")
        plt.xlabel(r"Budget ($\gamma_m B_m$)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    
    # def fit_q_vs_B(self, model="polynomial"):
    #     """
    #     Fit q values as a function of B for each user type and round.

    #     Parameters:
    #         model (str): The type of model to fit. Options: 'linear', 'polynomial', 'logistic', 'exp_decay'.

    #     Returns:
    #         list: A nested list of fitted models. Shape: (I, T)
    #     """
    #     I, T = self.opt_problem.I, self.opt_problem.T
    #     fitted_models = [[None for _ in range(T)] for _ in range(I)]  # Initialize nested list

    #     for i in range(I):
    #         for t in range(T):
    #             B = np.array(self.B_values)
    #             q = np.array([q_matrix[i, t] for q_matrix in self.q_values])

    #             # Fit the model
    #             if model == "linear":
    #                 coeffs = np.polyfit(B, q, 1)
    #                 fitted_models[i][t] = np.poly1d(coeffs)
    #             elif model == "polynomial":
    #                 coeffs = np.polyfit(B, q, 2)
    #                 fitted_models[i][t] = np.poly1d(coeffs)
    #             elif model == "logistic":
    #                 try:
    #                     params, _ = curve_fit(self.logistic, B, q, p0=[max(q), 0.01, np.median(B)])
    #                     fitted_models[i][t] = lambda B_new, p=params: self.logistic(B_new, *p)
    #                 except RuntimeError:
    #                     print(f"Failed to fit logistic for user {i+1}, round {t+1}")
    #                     fitted_models[i][t] = None
    #             elif model == "exp_decay":
    #                 # Scale B for better fitting stability
    #                 B_scaled = B / max(B)
    #                 try:
    #                     params, _ = curve_fit(
    #                         self.exp_decay, B_scaled, q, p0=[max(q), 1e-4], maxfev=10000
    #                     )
    #                     fitted_models[i][t] = lambda B_new, p=params: self.exp_decay(
    #                         B_new / max(B), *p
    #                     )
    #                 except RuntimeError:
    #                     print(f"Failed to fit exp_decay for user {i+1}, round {t+1}")
    #                     fitted_models[i][t] = None  # Handle fit failure gracefully
    #             else:
    #                 raise ValueError(f"Unsupported model type: {model}")

    #     return fitted_models

    # def predict_q(self, fitted_models, B_new):
    #     """
    #     Predict q values for a given budget (B_new) using the fitted models.

    #     Parameters:
    #         fitted_models (list): A nested list of fitted models. Shape: (I, T)
    #         B_new (float or array-like): A specific budget value or range to predict q values for.

    #     Returns:
    #         np.ndarray: Predicted q values. Shape: (I, T, len(B_new)) if B_new is a range.
    #     """
    #     I, T = self.opt_problem.I, self.opt_problem.T
    #     q_pred = np.zeros((I, T, len(B_new))) if isinstance(B_new, (list, np.ndarray)) else np.zeros((I, T))

    #     for i in range(I):  # Iterate over user types
    #         for t in range(T):  # Iterate over rounds
    #             model = fitted_models[i][t]
    #             if model is not None:
    #                 q_pred[i, t] = model(B_new)  # Predict q for the given B_new
    #             else:
    #                 q_pred[i, t] = np.nan  # Handle missing models gracefully

    #     return q_pred

    # def plot_fitted_q_vs_B(self, fitted_models, B_new):
    #     """
    #     Plot the fitted q values vs. budget (B), grouped by user type.

    #     Parameters:
    #         fitted_models (list): A list of fitted models for each user type and round.
    #                               Shape: (I, T)
    #         B_new (array-like): A range of B values to predict q values.
    #     """
    #     I, T = self.opt_problem.I, self.opt_problem.T

    #     for i in range(I):  # Iterate over user types
    #         plt.figure(figsize=(10, 7))
    #         for t in range(T):  # Iterate over rounds for each user type
    #             # Original data
    #             q_t_i = [q[i, t] for q in self.q_values]

    #             # Get the model for this user type and round
    #             model = fitted_models[i][t]
    #             if model is not None:
    #                 # Predict q values for the new B range
    #                 q_pred = model(B_new)

    #                 # Plot original data
    #                 plt.scatter(self.B_values, q_t_i, label=f"Observed Round {t+1}", zorder=5)
    #                 # Plot the predictions
    #                 plt.plot(
    #                     B_new, q_pred, linestyle="-", label=rf"$q_{{{i+1}}}^{{{t+1}}}$"
    #                 )

    #         # Customize plot
    #         plt.title(f"Fitted $q_{{{i+1}}}^t$ vs. Budget (B) for User Type {i+1}")
    #         plt.xlabel("Budget (B)")
    #         plt.ylabel(rf"$q_{{{i+1}}}^t$")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
