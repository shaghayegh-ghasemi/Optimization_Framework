import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

class StackelbergSolver:
    def __init__(self, fitted_models, M, xi = 500, c = 0.3, epsilon=1e-4):
        """
        Initialize StackelbergSolver with fitted model functions.
        
        Parameters:
            fitted_models (list): List of fitted model functions for each cluster.
            M (int): Number of clusters.
        """
        self.fitted_models = fitted_models  # List of fitted functions for accuracy
        self.M = M  # Number of local servers
        self.xi = xi
        self.c = c
        self.epsilon = epsilon
        
    def system_of_equations(self, variables, T):
        """
        Constructs the system of 2M nonlinear equations.
        
        Parameters:
            variables (numpy array): Contains M gamma values followed by M accuracy values.
            T (float): Total budget allocated by the main server.
        
        Returns:
            list: System of equations to solve for gamma and accuracy.
        """
        gamma = np.clip(variables[:self.M], 1e-3, 1-1e-3)  # Ensure gamma remains within a valid range
        A = np.clip(variables[self.M:], 1e-3, None)  # Ensure accuracy remains positive
        equations = []
        
        # M equations from the fitted models
        total_A = max(np.sum(A), 1e-6)
        for m in range(self.M):
            scaled_budget = gamma[m] * (A[m] / total_A) * T  # Avoid division instability
            fitted_accuracy = self.fitted_models[m](scaled_budget)
            # print(f"Cluster: {m}, Fitted Accuracy: {fitted_accuracy}")
            equations.append(A[m] - fitted_accuracy)
            
        # M equations from Equation (27) with stability fix
        sum_A_except_m = np.maximum(np.sum(A) - A, 1e-6)  # Avoid zero division
        for m in range(self.M):
            exp_term = np.exp(np.clip(self.c * sum_A_except_m[m], -100, 100))  # Prevent overflow
            denom = max((1 - gamma[m]) - exp_term, 1e-6)  # Ensure denominator is not too small
            rhs = (sum_A_except_m[m] * exp_term) / denom
            equations.append(A[m] - rhs)
        
        return equations
    
    def solve_system(self, T):
        """
        Solves the system of equations for given T using least_squares with constraints.
        
        Parameters:
            T (float): Total budget allocated by the main server.
        
        Returns:
            numpy array: Solutions for gamma and A values.
        """
        initial_A_guess = np.random.uniform(0.8, 1.2, self.M) * max(T / (10 * self.M), 1.0)  # Slight variation in A
        initial_gamma_guess = np.random.uniform(0.4, 0.6, self.M)  # Randomize gamma in a reasonable range
        initial_guess = np.concatenate((initial_gamma_guess, initial_A_guess))
        bounds = ([1e-3] * self.M + [1e-3] * self.M, [1-1e-3] * self.M + [np.inf] * self.M)  # Constraints
        solution = least_squares(self.system_of_equations, initial_guess, bounds=bounds, args=(T,))
        return solution.x
    
    def solve_for_T_values(self, T_values):
        """
        Solves the system for multiple values of T and prints the results.
        
        Parameters:
            T_values (list): List of T values to solve the system for.
        """
        for T in T_values:
            solution = self.solve_system(T)
            gamma_values = solution[:self.M]
            accuracy_values = solution[self.M:]
            print(f"For T = {T}:")
            print(f"  Gamma values: {gamma_values}")
            print(f"  Accuracy values: {accuracy_values}\n")
            
    def compute_dA_dT(self, T):
        """
        Computes numerical derivative dA/dT using central differences for better accuracy.
        """
        step = max(self.epsilon, 1.0)  # Increase step size for smoother differentiation
        A_T_plus = self.solve_system(T + step)[self.M:]
        A_T_minus = self.solve_system(T - step)[self.M:]
        dA_dT = (A_T_plus - A_T_minus) / (2 * step)  # Central difference
        return dA_dT, (A_T_plus + A_T_minus) / 2  # Return averaged A_T for stability
    
    def main_server_utility(self, T):
        _, A_T = self.compute_dA_dT(T)
        print(f"Debug: For T={T}, A_T={A_T}")
        return self.xi * np.log(1 + np.sum(A_T)) - T
    
    def main_server_derivative(self, T):
        dA_dT, A_T = self.compute_dA_dT(T)
        numerator = np.sum(dA_dT)
        denominator = max(1 + np.sum(A_T), 1e-6)  # Prevent division by near-zero values
        return (self.xi * numerator / denominator) - 1
    
    def find_optimal_T(self, T_min=1000, T_max=20000):
        """
        Finds the optimal T by solving dPi/dT = 0 using scalar minimization.
        If dPi/dT is far from zero, it retries with adjusted bounds.
        """
        print("Plotting Utility Function for Debugging...")
        self.plot_utility_and_derivative(T_min, T_max)
        
        result = minimize_scalar(lambda T: abs(self.main_server_derivative(T)), bounds=(T_min, T_max), method='bounded')
        dPi_dT_opt = self.main_server_derivative(result.x)
        print(f"Verifying Optimal T: dPi/dT({result.x}) = {dPi_dT_opt}")
        
        if abs(dPi_dT_opt) > 1e-2:
            print("Warning: dPi/dT is not close to zero! Adjusting optimization bounds.")
            return self.find_optimal_T(T_min=result.x * 0.8, T_max=result.x * 1.2)  # Refine search range
        
        return result.x
    
    def plot_utility_and_derivative(self, T_min, T_max):
        """
        Plots the main server utility function and its derivative over a range of T.
        """
        T_values = np.linspace(T_min, T_max, 50)
        Pi_values = [self.main_server_utility(T) for T in T_values]
        dPi_values = [self.main_server_derivative(T) for T in T_values]
        
        plt.figure(figsize=(10, 5))
        
        # Plot Utility Function
        plt.subplot(1, 2, 1)
        plt.plot(T_values, Pi_values, marker='o', linestyle='-')
        plt.xlabel("Total Budget (T)")
        plt.ylabel("Main Server Utility Pi(T)")
        plt.title("Utility Function of the Main Server")
        plt.grid()
        
        # Plot Derivative
        plt.subplot(1, 2, 2)
        plt.plot(T_values, dPi_values, marker='s', linestyle='-', color='r')
        plt.xlabel("Total Budget (T)")
        plt.ylabel("Derivative dPi/dT")
        plt.title("Derivative of Main Server Utility")
        plt.grid()
        
        plt.tight_layout()
        plt.show()
        
    def plot_utility(self, T_min, T_max):
        """
        Plots the main server utility function over a range of T to check concavity.
        """
        T_values = np.linspace(T_min, T_max, 50)
        Pi_values = [self.main_server_utility(T) for T in T_values]
        
        plt.figure(figsize=(8, 5))
        plt.plot(T_values, Pi_values, marker='o', linestyle='-')
        plt.xlabel("Total Budget (T)")
        plt.ylabel("Main Server Utility Pi(T)")
        plt.title("Utility Function of the Main Server")
        plt.grid()
        plt.show()