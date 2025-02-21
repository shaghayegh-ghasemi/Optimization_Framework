import numpy as np
from scipy.optimize import least_squares, root_scalar, curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt



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
            B_m = gamma[m] * B_m
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
            # A_m_value = np.exp(log_A_m)  # Convert back to original scale
            equation_value = A[m] - log_A_m
            equations.append(equation_value)
            
            # print(f"Eq (27): m = {m}, sum_A_except_m = {sum_A_except_m:.5e}, log_A_m = {log_A_m:.5e}, A_m_value = {A_m_value:.5e}, denom = {denom:.5e}, equation_value = {equation_value:.5e}")

        return np.array(equations)

    def parametric_solution(self, T_values, initial_guess=None, max_retries=3):
        """
        Solve the system of 2M equations for a range of T values to derive A_m(T) and gamma_m(T) with retries for stability.
        
        Parameters:
            T_values (list or np.array): List of T values for which to solve the system.
            initial_guess (np.array, optional): Initial guess for [transformed_A_1, ..., transformed_A_M, transformed_gamma_1, ..., transformed_gamma_M].
            max_retries (int): Maximum number of retries if the solver fails to converge.

        Returns:
            dict: Dictionary with T as keys and solutions [A_1, ..., A_M, gamma_1, ..., gamma_M] as values.
        """
        if initial_guess is None:
            initial_guess = np.concatenate((10 * np.ones(self.M), 0.5 * np.ones(self.M)))  # Start with transformed_A = 0 and transformed_gamma = 0 (A=1, gamma=0.5)

        solutions = {}
        for T in T_values:
            success = False
            retries = 0
            while not success and retries < max_retries:
                try:
                    res = least_squares(self.system_of_equations, initial_guess, args=(T,))
                    if res.success:
                        transformed_solution = res.x
                        success = True
                    else:
                        print(f"Warning: Solver did not converge for T={T}. Retrying ({retries + 1}/{max_retries})...")
                        initial_guess = np.random.rand(len(initial_guess)) * 0.1  # New random guess for retry
                        retries += 1
                except Exception as e:
                    print(f"Solver failed for T={T}. Error: {e}. Retrying ({retries + 1}/{max_retries})...")
                    initial_guess = np.random.rand(len(initial_guess)) * 0.1
                    retries += 1

            if success:
                # Transform back to original A and gamma
                A_solution = np.exp(transformed_solution[:self.M])
                gamma_solution = 1 / (1 + np.exp(-transformed_solution[self.M:]))
                
                # Store the solution
                solutions[T] = np.concatenate((A_solution, gamma_solution))
                initial_guess = transformed_solution  # Use the solution as the next initial guess for stability

                print(f"T={T:.2f}: A_solution={A_solution}, gamma_solution={gamma_solution}")
            else:
                print(f"Failed to find a solution for T={T} after {max_retries} retries.")
        
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
        Find the optimal T by solving \( \frac{\partial \Pi(T)}{\partial T} = 0 \) using root-finding methods.
        """

        # Step 1: Compute the partial derivative of A_m with respect to T for each cluster
        d_A_m_d_T = []
        for m in range(self.M):
            A_m_values = np.array([solutions[T][m] for T in T_values])
            d_A_m = self.numerical_derivative(A_m_values, T_values)
            d_A_m_d_T.append(d_A_m)

        d_A_m_d_T = np.array(d_A_m_d_T)

        # Step 2: Compute numerator
        numerator = self.xi * np.sum(d_A_m_d_T, axis=0)

        # Step 3: Compute denominator
        sum_A_values = np.sum([solutions[T][:self.M] for T in T_values], axis=1)
        denominator = 1 + sum_A_values

        # Step 4: Compute d_Pi(T)
        d_Pi_T = (numerator / denominator) - 1

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

        # Step 5: Find the root of d_Pi_T = 0 using Brent's method
        try:
            result = root_scalar(lambda T: np.interp(T, T_values, d_Pi_T),
                                bracket=[T_values[0], T_values[-1]], method='brentq')
            if result.converged:
                optimal_T = result.root
                print(f"Optimal T found using root_scalar: {optimal_T}")
                print(f"d_Pi(T) at optimal T: {np.interp(optimal_T, T_values, d_Pi_T):.5e}")
                return optimal_T
            else:
                print("root_scalar did not converge.")
                return None
        except ValueError:
            print("Root-finding failed. No sign change detected in the given range.")
            return None



