import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # Stable implementation of sigmoid
from scipy.stats import linregress
import matplotlib.pyplot as plt

class Fitter:
    def __init__(self, results, opt_problem):
        """
        Initialize QFitter with results and optimization problem.

        Parameters:
            results (list): Results containing B-values, q-values, and other outputs.
            opt_problem (object): Optimization problem object defining I (user types) and T (rounds).
        """
        self.results = results
        self.opt_problem = opt_problem
        self.B_values = [result[0] for result in results]  # Extract B-values
        self.q_values = [result[2] for result in results]  # Extract q matrices
        self.total_accuracies = [result[5] for result in results]  # Extract total accuracies

    @staticmethod
    def logistic(B, L, k, B0):
        """Logistic growth function with numerical stability."""
        z = -k * (B - B0)
        return L * expit(-z)  # Use expit for stable logistic sigmoid

    @staticmethod
    def exp_decay(B, L, k):
        """Exponential growth with asymptote."""
        # Clip B to prevent overflow
        B = np.clip(B, 0, 1e3)
        return L * (1 - np.exp(-k * B))
    
    def generalized_logistic(self, B, A_max, k, B_0, v):
        """ Generalized Logistic Function (Richards Curve). """
        return A_max / (1 + np.exp(-k * (B - B_0))) ** v


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
    
    def fit_q_vs_B(self, model="polynomial"):
        """
        Fit q values as a function of B for each user type and round.

        Parameters:
            model (str): The type of model to fit. Options: 'linear', 'polynomial', 'logistic', 'exp_decay'.

        Returns:
            list: A nested list of fitted models. Shape: (I, T)
        """
        I, T = self.opt_problem.I, self.opt_problem.T
        fitted_models = [[None for _ in range(T)] for _ in range(I)]  # Initialize nested list

        for i in range(I):
            for t in range(T):
                B = np.array(self.B_values)
                q = np.array([q_matrix[i, t] for q_matrix in self.q_values])

                # Fit the model
                if model == "linear":
                    coeffs = np.polyfit(B, q, 1)
                    fitted_models[i][t] = np.poly1d(coeffs)
                elif model == "polynomial":
                    coeffs = np.polyfit(B, q, 2)
                    fitted_models[i][t] = np.poly1d(coeffs)
                elif model == "logistic":
                    try:
                        params, _ = curve_fit(self.logistic, B, q, p0=[max(q), 0.01, np.median(B)])
                        fitted_models[i][t] = lambda B_new, p=params: self.logistic(B_new, *p)
                    except RuntimeError:
                        print(f"Failed to fit logistic for user {i+1}, round {t+1}")
                        fitted_models[i][t] = None
                elif model == "exp_decay":
                    # Scale B for better fitting stability
                    B_scaled = B / max(B)
                    try:
                        params, _ = curve_fit(
                            self.exp_decay, B_scaled, q, p0=[max(q), 1e-4], maxfev=10000
                        )
                        fitted_models[i][t] = lambda B_new, p=params: self.exp_decay(
                            B_new / max(B), *p
                        )
                    except RuntimeError:
                        print(f"Failed to fit exp_decay for user {i+1}, round {t+1}")
                        fitted_models[i][t] = None  # Handle fit failure gracefully
                else:
                    raise ValueError(f"Unsupported model type: {model}")

        return fitted_models

    def predict_q(self, fitted_models, B_new):
        """
        Predict q values for a given budget (B_new) using the fitted models.

        Parameters:
            fitted_models (list): A nested list of fitted models. Shape: (I, T)
            B_new (float or array-like): A specific budget value or range to predict q values for.

        Returns:
            np.ndarray: Predicted q values. Shape: (I, T, len(B_new)) if B_new is a range.
        """
        I, T = self.opt_problem.I, self.opt_problem.T
        q_pred = np.zeros((I, T, len(B_new))) if isinstance(B_new, (list, np.ndarray)) else np.zeros((I, T))

        for i in range(I):  # Iterate over user types
            for t in range(T):  # Iterate over rounds
                model = fitted_models[i][t]
                if model is not None:
                    q_pred[i, t] = model(B_new)  # Predict q for the given B_new
                else:
                    q_pred[i, t] = np.nan  # Handle missing models gracefully

        return q_pred

    def plot_fitted_q_vs_B(self, fitted_models, B_new):
        """
        Plot the fitted q values vs. budget (B), grouped by user type.

        Parameters:
            fitted_models (list): A list of fitted models for each user type and round.
                                  Shape: (I, T)
            B_new (array-like): A range of B values to predict q values.
        """
        I, T = self.opt_problem.I, self.opt_problem.T

        for i in range(I):  # Iterate over user types
            plt.figure(figsize=(10, 7))
            for t in range(T):  # Iterate over rounds for each user type
                # Original data
                q_t_i = [q[i, t] for q in self.q_values]

                # Get the model for this user type and round
                model = fitted_models[i][t]
                if model is not None:
                    # Predict q values for the new B range
                    q_pred = model(B_new)

                    # Plot original data
                    plt.scatter(self.B_values, q_t_i, label=f"Observed Round {t+1}", zorder=5)
                    # Plot the predictions
                    plt.plot(
                        B_new, q_pred, linestyle="-", label=rf"$q_{{{i+1}}}^{{{t+1}}}$"
                    )

            # Customize plot
            plt.title(f"Fitted $q_{{{i+1}}}^t$ vs. Budget (B) for User Type {i+1}")
            plt.xlabel("Budget (B)")
            plt.ylabel(rf"$q_{{{i+1}}}^t$")
            plt.legend()
            plt.grid(True)
            plt.show()

    def fit_accuracy_vs_B(self, model="logistic"):
        """
        Fit the summation of accuracies (A_m) as a function of B.

        Parameters:
            model (str): The type of model to fit ('logistic', 'exp_decay', 'generalized_logistic', 'piecewise').

        Returns:
            function: Fitted model function.
        """
        B_values = np.array(self.B_values)
        A_m_values = np.array(self.total_accuracies)

        if model == "logistic":
            params, _ = curve_fit(self.logistic, B_values, A_m_values, p0=[max(A_m_values), 0.01, np.median(B_values)])
            fitted_model = lambda B_new: self.logistic(B_new, *params)

        elif model == "exp_decay":
            params, _ = curve_fit(self.exp_decay, B_values, A_m_values, p0=[max(A_m_values), 1e-4], maxfev=10000)
            fitted_model = lambda B_new: self.exp_decay(B_new, *params)

        elif model == "generalized_logistic":
            params, _ = curve_fit(self.generalized_logistic, B_values, A_m_values, p0=[max(A_m_values), 0.01, np.median(B_values), 1])
            fitted_model = lambda B_new: self.generalized_logistic(B_new, *params)

        elif model == "piecewise":
            # First, fit a linear model for the small-budget region
            B_critical = np.median(B_values)  # Choose dynamically based on median
            small_B_mask = B_values < B_critical
            slope, intercept, _, _, _ = linregress(B_values[small_B_mask], A_m_values[small_B_mask])

            # Fit the logistic model for large-budget region
            params, _ = curve_fit(self.logistic, B_values, A_m_values, p0=[max(A_m_values), 0.01, np.median(B_values)])
            
            # Combine models
            fitted_model = lambda B_new: self.piecewise_linear_logistic(B_new, slope, intercept, *params)

        else:
            raise ValueError(f"Unsupported model type: {model}")

        return fitted_model

    def plot_fitted_total_accuracy(self, fitted_model, B_new):
        """
        Plot the fitted total accuracy (A_m) vs. Budget (B).

        Parameters:
            fitted_model (function): The fitted model for A_m.
        """
        B_values = np.array(self.B_values)
        A_m_values = np.array(self.total_accuracies)

        # Generate predictions for a finer B range
        # B_new = np.linspace(min(B_values), max(B_values), 100)
        A_m_pred = fitted_model(B_new)

        # Plot observed and fitted values
        plt.figure(figsize=(10, 7))
        plt.scatter(B_values, A_m_values, color='b', label="Observed $A_m$")
        plt.plot(B_new, A_m_pred, color='r', linestyle='-', label="Fitted Curve")
        plt.title("Summation of Accuracies ($A_m$) vs. Budget (B)")
        plt.xlabel("Budget (B)")
        plt.ylabel("Total Accuracy ($A_m$)")
        plt.legend()
        plt.grid(True)
        plt.show()