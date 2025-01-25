import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class QFitter:
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

    def logistic(self, B, L, k, B0):
        """Logistic growth function."""
        return L / (1 + np.exp(-k * (B - B0)))

    def exp_decay(self, B, L, k):
        """Exponential growth with asymptote."""
        # Clip B to prevent overflow
        B = np.clip(B, 0, 1e3)
        return L * (1 - np.exp(-k * B))

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
