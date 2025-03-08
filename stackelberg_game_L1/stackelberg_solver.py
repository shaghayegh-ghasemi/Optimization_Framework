import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

class StackelbergSolver:
    def __init__(self, fitted_models, M, xi = 1000.0, c = 0.3, epsilon=1e-4):
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
        total_A = max(np.sum(A), T / 10)  # Ensure a minimum meaningful scale

        # B = np.maximum(gamma * (A / total_A) * T, 1e-3)  # Avoid very small B values
        # weights = np.linspace(0.8, 1.2, self.M)  # Slight perturbation for diversity
        # B = np.maximum(gamma * (A / total_A) * T * weights, 1e-3)
        B = np.maximum(gamma * (A / total_A) * T * np.linspace(0.8, 1.2, self.M), 1e-3)

        fitted_accuracies = np.array([self.fitted_models[m](B[m]) for m in range(self.M)])
        for m in range(self.M):
            # print(f"Debug: T={T}, Cluster {m}, B_m={B[m]}, A_m(expected)={fitted_accuracies[m]}, A_m(actual)={A[m]}")  # Debug accuracy values
            equations.append(A[m] - fitted_accuracies[m])
            
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
        # Improved Initial Guess for A_m
        # initial_A_guess = np.full(self.M, max(T / (2 * self.M), 1e-2))  # Set A_m to a fraction of T
        initial_A_guess = np.full(self.M, max(T / (2 * self.M), 1e-2)) * np.linspace(0.9, 1.1, self.M)

        # Improved Initial Guess for gamma_m
        initial_gamma_guess = np.random.uniform(0.45, 0.55, self.M)  # Start with balanced allocation

        # Combine initial guesses
        initial_guess = np.concatenate((initial_gamma_guess, initial_A_guess))

        # Improved Bounds
        gamma_lower_bound = [1e-4] * self.M
        gamma_upper_bound = [1 - 1e-4] * self.M
        A_lower_bound = [max(T / (50 * self.M), 1e-3)] * self.M  # Ensure A_m doesn't collapse
        A_upper_bound = [np.inf] * self.M  # No upper bound restriction

        bounds = (gamma_lower_bound + A_lower_bound, gamma_upper_bound + A_upper_bound)

        # Solve System
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
        step = max(self.epsilon, min(T * 0.01, 10))  # Step size scales with T
        A_T_plus = self.solve_system(T + step)[self.M:]
        A_T_minus = self.solve_system(T - step)[self.M:]
        dA_dT = np.clip((A_T_plus - A_T_minus) / (2 * step), -100, 100)
        return dA_dT, (A_T_plus + A_T_minus) / 2  # Return averaged A_T for stability
    
    def main_server_utility(self, T):
        _, A_T = self.compute_dA_dT(T)
        A_T_sum = np.sum(A_T)
        
        # Induce concavity by limiting growth
        log_term = np.log(1 + np.sum(A_T)) * (1 - np.exp(-T / 1200))
        
        return self.xi * log_term - T

    def main_server_derivative(self, T):
        dA_dT, A_T = self.compute_dA_dT(T)
        numerator = np.sum(dA_dT)
        denominator = max(1 + np.sum(A_T), 1e-6)  # Prevent division by near-zero values
        return (self.xi * numerator / denominator) - 1
    
    def find_optimal_T(self, T_min=1000, T_max=5000):
        """
        Finds the optimal T by maximizing the main server utility function directly.
        This ensures we get the true peak instead of relying on numerical derivatives.
        """
        print("Plotting Utility Function for Debugging...")
        self.plot_utility_and_derivative(T_min, T_max)  # Debugging visualization

        # **1st Attempt: Direct Maximization**
        result = minimize_scalar(lambda T: -self.main_server_utility(T), bounds=(T_min, T_max), method='bounded')

        # **Check if the found T is reasonable**
        optimal_T = result.x
        print(f"Initial optimization found T = {optimal_T}, with Pi(T) = {self.main_server_utility(optimal_T)}")

        # **2nd Attempt: Grid Search in Refined Range**
        if not (T_min <= optimal_T <= T_max):  
            print("Warning: Optimization might have failed. Running Grid Search...")

        T_values = np.linspace(T_min, T_max, 100)  # Search in a finer range
        Pi_values = [self.main_server_utility(T) for T in T_values]
        optimal_T_grid = T_values[np.argmax(Pi_values)]

        print(f"Grid search found T = {optimal_T_grid}, with Pi(T) = {max(Pi_values)}")

        return optimal_T_grid  # Return the best T from grid search

    
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