import numpy as np
from scipy.optimize import least_squares, brentq, minimize_scalar
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



class StackelbergSolver:
    def __init__(self, fitted_models, M):
        """
        Initialize StackelbergSolver with fitted model functions.
        
        Parameters:
            fitted_models (list): List of fitted model functions for each cluster.
            M (int): Number of clusters.
        """
        self.fitted_models = fitted_models  # List of fitted functions for accuracy
        self.M = M  # Number of local servers
        self.xi = 1

    def system_of_equations(self, transformed_vars, T):
        """
        Define the system of 2M equations with log/sigmoid transformations and stabilized equation (27).

        Parameters:
            transformed_vars (np.array): Array containing [transformed_A_1, ..., transformed_A_M, transformed_gamma_1, ..., transformed_gamma_M].
            T (float): Total budget T.

        Returns:
            np.array: Array of 2M equations.
        """
        # Apply transformations to ensure A > 0 and 0 < gamma < 1
        A = np.exp(transformed_vars[:self.M])  # Ensure A > 0
        gamma = 1 / (1 + np.exp(-transformed_vars[self.M:]))  # Ensure 0 < gamma < 1

        c = 1
        
        equations = []

        # Equation (9): Ensure A_m matches the estimated value from the fitted model
        for m in range(self.M):
            B_m = (A[m] / np.sum(A)) * T
            A_m_estimated = self.fitted_models[m](B_m)
            residual_A = A_m_estimated - A[m]
            equations.append(residual_A)
            # print(f"Eq (9): m = {m}, A[{m}] = {A[m]:.5e}, B_m = {B_m:.5e}, A_m_estimated = {A_m_estimated:.5e}, residual_A = {residual_A:.5e}")

        # Equation (27): Stabilized recursive relationship for A_m and gamma_m
        for m in range(self.M):
            sum_A_except_m = np.sum(A) - A[m]
            log_exp_term = np.clip(c * sum_A_except_m, -500, 500)  # Stabilized logarithmic computation
            exp_term = np.exp(log_exp_term)
            denom = (1 - gamma[m]) - exp_term
            if denom <= 1e-10:
                # print(f"Warning: Denominator too small for m = {m}, clipping to avoid division by zero.")
                denom = 1e-5  # Avoid division by zero

            # Stabilized calculation using logarithmic approach
            log_A_m = np.log(sum_A_except_m) + log_exp_term - np.log(np.abs(denom))
            A_m_value = np.exp(log_A_m)  # Convert back to original scale
            equation_value = A[m] - A_m_value
            equations.append(equation_value)
            
            # print(f"Eq (27): m = {m}, sum_A_except_m = {sum_A_except_m:.5e}, log_A_m = {log_A_m:.5e}, A_m_value = {A_m_value:.5e}, denom = {denom:.5e}, equation_value = {equation_value:.5e}")

        return np.array(equations)


    def parametric_solution(self, T_values, initial_guess=None):
        """
        Solve the system of 2M equations for a range of T values to derive A_m(T) and gamma_m(T).

        Parameters:
            T_values (list or np.array): List of T values for which to solve the system.
            initial_guess (np.array, optional): Initial guess for [transformed_A_1, ..., transformed_A_M, transformed_gamma_1, ..., transformed_gamma_M].

        Returns:
            dict: Dictionary with T as keys and solutions [A_1, ..., A_M, gamma_1, ..., gamma_M] as values.
        """
        if initial_guess is None:
            initial_guess = np.concatenate((np.zeros(self.M), np.zeros(self.M)))  # Start with transformed_A = 0 and transformed_gamma = 0 (A=1, gamma=0.5)

        solutions = {}
        for T in T_values:
            res = least_squares(self.system_of_equations, initial_guess, args=(T,))
            transformed_solution = res.x
            
            # Transform back to original A and gamma
            A_solution = np.exp(transformed_solution[:self.M])
            gamma_solution = 1 / (1 + np.exp(-transformed_solution[self.M:]))
            
            solutions[T] = np.concatenate((A_solution, gamma_solution))
            
            initial_guess = transformed_solution  # Use the solution as the next initial guess for stability

        return solutions


    
    def numerical_derivative(self, func_values, T_values):
        """
        Compute the numerical derivative using central differences.
        
        Parameters:
            func_values (np.array): Values of the function at corresponding T_values.
            T_values (np.array): T values.
        
        Returns:
            np.array: Numerical derivative of func_values with respect to T_values.
        """
        d_func = np.gradient(func_values, T_values)
        return d_func
        
    def find_optimal_T(self, T_values, solutions):
        """
        Find the optimal T by solving \( \frac{\partial \Pi(T)}{\partial T} = 0 \) using the correct formula with logging for debugging.

        Parameters:
            T_values (list or np.array): List of T values.
            solutions (dict): Dictionary of solutions [A_1, ..., A_M, gamma_1, ..., gamma_M] for each T.

        Returns:
            float: Optimal T value.
        """
        # Step 1: Compute the partial derivative of A_m with respect to T for each cluster
        d_A_m_d_T = []
        for m in range(self.M):
            A_m_values = np.array([solutions[T][m] for T in T_values])  # Extract A_m for each T
            d_A_m = self.numerical_derivative(A_m_values, T_values)  # Compute the numerical derivative of A_m with respect to T
            d_A_m_d_T.append(d_A_m)

            # Log values for debugging
            print(f"\nCluster {m + 1}/{self.M}:")
            print(f"A_m_values = {A_m_values}")
            print(f"d_A_m/d_T = {d_A_m}")

        d_A_m_d_T = np.array(d_A_m_d_T)  # Shape: (M, len(T_values))

        # Step 2: Compute the numerator of d_Pi_T
        numerator = self.xi * np.sum(d_A_m_d_T, axis=0)  # Sum the derivatives across clusters for each T
        print(f"\nNumerator (sum of d_A_m/d_T across all clusters): {numerator}")

        # Step 3: Compute the denominator (1 + sum of A_m for each T)
        sum_A_values = np.sum([solutions[T][:self.M] for T in T_values], axis=1)
        denominator = 1 + sum_A_values
        print(f"Sum of A_m for each T: {sum_A_values}")
        print(f"Denominator (1 + sum_A_values): {denominator}")

        # Step 4: Compute the derivative of Pi(T)
        d_Pi_T = numerator / denominator - 1
        print(f"d_Pi_T values: {d_Pi_T}")

        # Plot d_Pi_T for visualization
        plt.figure(figsize=(10, 6))
        plt.plot(T_values, d_Pi_T, label="d_Pi(T)", marker='o')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("T")
        plt.ylabel("d_Pi(T)")
        plt.title("Derivative of Pi(T) vs T")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Step 5: Find the root of d_Pi_T = 0
        if np.all(d_Pi_T < 0):
            print("d_Pi_T is negative for all T. Returning T_values[-1].")
            return T_values[-1]
        elif np.all(d_Pi_T > 0):
            print("d_Pi_T is positive for all T. Returning T_values[0].")
            return T_values[0]
        else:
            result = minimize_scalar(lambda T: abs(np.interp(T, T_values, d_Pi_T)), bounds=(T_values[0], T_values[-1]), method='bounded')
            optimal_T = result.x
            print(f"Optimal T found: {optimal_T}")
            print(f"d_Pi(T) at optimal T: {np.interp(optimal_T, T_values, d_Pi_T):.5e}")
            return optimal_T



