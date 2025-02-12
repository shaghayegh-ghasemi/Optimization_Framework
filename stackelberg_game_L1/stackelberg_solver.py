import numpy as np
from scipy.optimize import least_squares, brentq
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
        
    def system_of_equations(self, variables, T):
        """
        Define the system of 2M equations to solve for A_m and gamma_m.
        
        Parameters:
            variables (np.array): Array containing [A_1, ..., A_M, gamma_1, ..., gamma_M].
            T (float): Total budget T.
        
        Returns:
            np.array: Array of 2M equations.
        """
        A = np.clip(variables[:self.M], 1e-10, None)  # Ensure A > 1e-10
        gamma = np.clip(variables[self.M:], 1e-10, 1 - 1e-10)  # Ensure 0 < gamma < 1
        equations = []
        
        # Equation (9): Ensure A_m matches the estimated value from the fitted model
        for m in range(self.M):
            B_m = (A[m] / np.sum(A)) * T  # Calculate B_m based on A_m and T
            A_m_estimated = self.fitted_models[m](B_m)  # Get A_m from the fitted function
            equations.append(A_m_estimated - A[m])  # Ensure the estimated A_m matches the current A_m
        
        # Equation (27): Recursive relationship for A_m and gamma_m with improved numerical stability
        for m in range(self.M):
            sum_A_except_m = np.sum(A) - A[m]
            print(f"Iteration: m = {m}, A[{m}] = {A[m]}, gamma[{m}] = {gamma[m]}, sum_A_except_m = {sum_A_except_m}")

            # Use np.clip to prevent overflow when calculating the exponential
            safe_exp = np.exp(np.clip(sum_A_except_m, -500, 500))  # Clip to avoid overflow
            denom = (1 - gamma[m]) - safe_exp

            # Handle possible division by zero or near-zero values
            if np.abs(denom) < 1e-10:
                print(f"Warning: Denominator is very small for m = {m}. Clipping to avoid division by zero.")
                denom = 1e-10

            # Calculate the equation with safe handling
            equation_value = A[m] - (sum_A_except_m * safe_exp / denom)
            equations.append(equation_value)
            
        return np.array(equations)

    def parametric_solution(self, T_values, initial_guess=None):
        """
        Solve the system of 2M equations for a range of T values to derive A_m(T) and gamma_m(T).

        Parameters:
            T_values (list or np.array): List of T values for which to solve the system.
            initial_guess (np.array, optional): Initial guess for [A_1, ..., A_M, gamma_1, ..., gamma_M].

        Returns:
            dict: Dictionary with T as keys and solutions [A_1, ..., A_M, gamma_1, ..., gamma_M] as values.
        """
        if initial_guess is None:
            initial_guess = np.concatenate((np.ones(self.M), 0.5 * np.ones(self.M)))

        bounds = (np.concatenate((1e-10 * np.ones(self.M), 1e-10 * np.ones(self.M))), 
                np.concatenate((np.inf * np.ones(self.M), (1 - 1e-10) * np.ones(self.M))))

        solutions = {}
        for T in T_values:
            res = least_squares(self.system_of_equations, initial_guess, args=(T,), bounds=bounds)
            solution = res.x
            solutions[T] = solution
            initial_guess = solution  # Use the previous solution as the next initial guess for stability

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
        Find the optimal T by solving \( \frac{\partial \Pi(T)}{\partial T} = 0 \) using numerical derivatives.

        Parameters:
            T_values (list or np.array): List of T values.
            solutions (dict): Dictionary of solutions [A_1, ..., A_M, gamma_1, ..., gamma_M] for each T.

        Returns:
            float: Optimal T value.
        """
        import matplotlib.pyplot as plt
        from scipy.optimize import minimize_scalar

        # Extract A_m values for each T and calculate sum_A for each T
        A_values = np.array([solutions[T][:self.M] for T in T_values])
        sum_A_values = np.sum(A_values, axis=1)  # Sum of A_m for each T
        d_sum_A = self.numerical_derivative(sum_A_values, T_values)  # First derivative of sum_A_values
        d_sum_A_smooth = savgol_filter(d_sum_A, window_length=11, polyorder=3)


        # Define the derivative of Pi(T)
        d_Pi_T = (d_sum_A_smooth / (1 + sum_A_values)) - 1

        # Plot d_Pi_T for visualization
        plt.plot(T_values, d_Pi_T, label="d_Pi(T)")
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("T")
        plt.ylabel("d_Pi(T)")
        plt.title("Derivative of Pi(T) vs T")
        plt.legend()
        plt.show()

        # Check if there is a zero-crossing in d_Pi_T
        if np.all(d_Pi_T < 0):
            print("d_Pi_T is negative for all T. Returning T_values[-1].")
            return T_values[-1]
        elif np.all(d_Pi_T > 0):
            print("d_Pi_T is positive for all T. Returning T_values[0].")
            return T_values[0]
        else:
            # Use minimize_scalar to find the closest point to zero if no clean root is available
            result = minimize_scalar(lambda T: abs(np.interp(T, T_values, d_Pi_T)), bounds=(T_values[0], T_values[-1]), method='bounded')
            optimal_T = result.x
            print(f"Optimal T found: {optimal_T}, d_Pi(T) at optimal T: {np.interp(optimal_T, T_values, d_Pi_T)}")
            return optimal_T

