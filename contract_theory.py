import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

class ExperimentRunner:
    def __init__(self, opt_problem, B_values):
        """
        Initialize the ExperimentRunner with an optimization problem and a range of B values.
        
        Parameters:
        - opt_problem: Instance of the OptimizationContractTheory class.
        - B_values: List or array of B values to test.
        """
        self.opt_problem = opt_problem
        self.B_values = B_values
        self.results = []  # To store the results for plotting
        self.accuracy_values = []
        self.savings_values = []

    def run(self):
        for B in self.B_values:
            q, utility, accuracy, savings = self.opt_problem.solve(B)
            self.results.append((B, utility, q, accuracy, savings))
            self.accuracy_values.append(accuracy)
            self.savings_values.append(savings)

    def plot_all_savings(self):
        """
        Plot savings for all rounds in one plot with different colors.
        """
        B_values = self.B_values
        savings = np.array(self.savings_values)  # Shape: (len(B_values), T)

        plt.figure(figsize=(10, 7))
        for t in range(self.opt_problem.T):
            plt.plot(B_values, savings[:, t], marker='o', linestyle='-', label=f"Round {t+1}")
        
        plt.title("Savings vs. Budget (B)")
        plt.xlabel("Budget (B)")
        plt.ylabel("Savings")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_accuracies(self):
        """
        Plot accuracy for all user types and rounds in one plot with different colors.
        """
        B_values = self.B_values
        accuracy = np.array(self.accuracy_values)  # Shape: (len(B_values), I, T)

        plt.figure(figsize=(10, 7))
        for i in range(self.opt_problem.I):
            for t in range(self.opt_problem.T):
                plt.plot(B_values, accuracy[:, i, t], marker='o', linestyle='-', 
                         label=f"User Type {i+1}, Round {t+1}")
        
        plt.title("Accuracy vs. Budget (B)")
        plt.xlabel("Budget (B)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_results(self):
        """
        Plot the utility as a function of B.
        """
        B_values = [result[0] for result in self.results]
        utilities = [result[1] for result in self.results]

        plt.figure(figsize=(8, 6))
        plt.plot(B_values, utilities, marker='o', linestyle='-', color='b')
        plt.title("Utility vs. Budget (B)")
        plt.xlabel("Budget (B)")
        plt.ylabel("Server Utility")
        plt.grid(True)
        plt.show()

    def plot_q_trends(self):
        """
        Plot the trends of q values for each user type and round over different B values.
        """
        B_values = [result[0] for result in self.results]
        q_values = [result[2] for result in self.results]  # Extract q matrices

        I, T = self.opt_problem.I, self.opt_problem.T

        for i in range(I):
            for t in range(T):
                q_t_i = [q[i, t] for q in q_values]
                plt.figure(figsize=(8, 6))
                plt.plot(B_values, q_t_i, marker='o', linestyle='-')
                plt.title(rf"Trend of $q_{{{i+1}}}^{{{t+1}}}$ vs. Budget (B)")
                plt.xlabel("Budget (B)")
                plt.ylabel(rf"$q_{{{i+1}}}^{{{t+1}}}$ (Contribution)")
                plt.grid(True)
                plt.show()

    def plot_q_by_round(self):
        """
        Plot the trends of q values for all user types in the same plot, grouped by each round.
        """
        B_values = [result[0] for result in self.results]
        q_values = [result[2] for result in self.results]  # Extract q matrices

        I, T = self.opt_problem.I, self.opt_problem.T

        for t in range(T):
            plt.figure(figsize=(10, 7))
            for i in range(I):
                q_t_i = [q[i, t] for q in q_values]
                plt.plot(B_values, q_t_i, marker='o', linestyle='-', label=rf"$q_{{{i+1}}}^{{{t+1}}}$")
            
            plt.title(rf"Trends of $q^{{{t+1}}}_i$ for All User Types in Round {t+1}")
            plt.xlabel("Budget (B)")
            plt.ylabel(rf"$q^{{{t+1}}}_i$ (Contributions for Round {t+1})")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_q_by_type(self):
        """
        Plot the trends of q values for each user type over all rounds in the same plot.
        Each plot corresponds to a single user type and includes all rounds.
        """
        B_values = [result[0] for result in self.results]
        q_values = [result[2] for result in self.results]  # Extract q matrices

        I, T = self.opt_problem.I, self.opt_problem.T

        for i in range(I):
            plt.figure(figsize=(10, 7))
            for t in range(T):
                q_t_i = [q[i, t] for q in q_values]
                plt.plot(B_values, q_t_i, marker='o', linestyle='-', label=rf"$q_{{{i+1}}}^{{{t+1}}}$")
            
            plt.title(rf"Trends of $q_{{{i+1}}}^t$ Across All Rounds")
            plt.xlabel("Budget (B)")
            plt.ylabel(rf"$q_{{{i+1}}}^t$ (Contributions for User Type {i+1})")
            plt.legend()
            plt.grid(True)
            plt.show()

    def display_results(self):
        """
        Display detailed results, including q values for each B.
        """
        for result in self.results:
            B, utility, q = result
            print(f"Budget (B): {B}")
            print(f"Utility: {utility}")
            print(f"Optimal q values:\n{q}")
            print("-" * 40)


class OptimizationContractTheory:
    def __init__(self, I, T, N, sigma, eta, p, theta, q_max):
        # Parameters
        self.I = I  # User types
        self.T = T  # Rounds
        self.N = N  # Total number of users
        self.sigma = sigma  # Calibration constants
        self.eta = eta  # Calibration constants
        self.p = p  # Probabilities for user types
        self.theta = theta  # Computation costs for user types
        self.q_max = q_max  # Maximum contribution for each round

        # Decision variable
        self.q = cp.Variable((I, T), nonneg=True)  # Contribution for each user type and round

        # Symbolic parameter
        self.B = cp.Parameter()  # Total budget parameter (passed later)

    def compute_reward(self):
        """
        Compute the reward for each user type i at each round t:
        R_i^t = θ_i * q_i^t + Δ_i, where Δ_i depends on q.
        """
        big_delta = self.compute_big_delta()
        R = [] 
        for t in range(self.T):
            R_t = []
            for i in range(self.I):
                R_t.append(self.theta[i] * self.q[i, t] + big_delta[i, t])
            R.append(cp.vstack(R_t))

        return cp.hstack(R)
    
    def compute_big_delta(self):
        """
        Compute Δ_i as a function of θ and q:
        Δ_i = Σ_{j=i+1}^{I} (θ_j - θ_{j-1}) * q_j^t
        """
        big_delta = []

        for t in range(self.T):
            big_delta_t = []
            for i in range(self.I - 1):
                big_delta_t.append(
                    sum(
                        (self.theta[j] - self.theta[j - 1]) * self.q[j, t]
                        for j in range(i + 1, self.I)
                    )
                )
            big_delta_t.append(0)  # Last user type has no Δ
            big_delta.append(cp.vstack(big_delta_t))
        return cp.hstack(big_delta)

    def compute_B_t(self):
        """
        Compute available budgets B_t based on cumulative rewards.
        """
        cumulative_R = cp.sum(cp.multiply(self.p[:, None] * self.N, self.compute_reward()), axis=0) # Shape (T,)
        # Calculate α_t (it should be a vector of length T)
        alpha_t = np.array([self.q_max[t] / np.sum(self.q_max[t:]) for t in range(self.T)]) 

        B_t = [] # This will store the budget values for each round t
        for t in range(self.T):
            if t == 0:
                B_t.append(alpha_t[t] * self.B)
            else:
                B_t.append(alpha_t[t] * (self.B - cp.sum(cumulative_R[:t])))

        return cp.hstack(B_t), cumulative_R # This should return a vector of shape (T,) for B_t

    def compute_savings(self, B_t):
        """
        Compute savings for each round as B_t - total payment for that round.
        """
        rewards = self.compute_reward()
        payments_t = cp.sum(cp.multiply(self.p[:, None] * self.N, rewards), axis=0)  # Total payments for each round
        savings = B_t - payments_t  # Savings for each round
        return savings, payments_t
    
    def compute_server_utility(self):
        """
        Compute server utility and accuracy values for all rounds and types.
        """
        reward = self.compute_reward()
        accuracy = cp.log(1 + self.eta * self.q)  # Shape: (I, T)
        utility = cp.sum(
            cp.multiply(self.p[:, None] * self.N, self.sigma * accuracy - reward)
        )
        return -1 * utility, accuracy

    def solve(self, B_value):
        """
        Solve the optimization problem and return q, utility, accuracy, and savings.
        """
        self.B.value = B_value

        # Compute dependent variables
        B_t, cumulative_R = self.compute_B_t()  # Extract both elements
        savings, payments_t = self.compute_savings(B_t)  # Pass only B_t
        utility, accuracy = self.compute_server_utility()

        # Constraints
        constraints = [self.q[:, t] <= self.q_max[t] for t in range(self.T)]
        constraints += [payments_t[t] <= B_t[t] for t in range(self.T)]
        constraints += [self.q >= 0]

        # Solve problem
        problem = cp.Problem(cp.Minimize(utility), constraints)
        problem.solve()

        return self.q.value, utility.value, accuracy.value, savings.value

        
        
if __name__ == '__main__':
    # Parameters
    I = 5  # User types
    T = 3  # Rounds
    N = 20 # Total number of users
    sigma = 1000 # Calibration constants
    eta = 1 # Calibration constants

    p = np.random.dirichlet(np.ones(I))  # the ith elements represents the probability that users belongs to types 𝑖 among the 𝑁 available users
    theta = np.sort(np.random.uniform(0.1, 0.9, I))  # Computation costs
    q_max = np.array([100 + i * 50 for i in range(T)]) # q_max at round 1 = 100, it's increading by 50 units round by round

    # Create an instance of the optimization problem
    opt_problem = OptimizationContractTheory(I, T, N, sigma, eta, p, theta, q_max)

    # Define a range of B values for testing
    B_values = np.linspace(500, 5000, 10)  # Example: Test B from 500 to 5000 in 10 steps

    # Create and run the experiment
    experiment = ExperimentRunner(opt_problem, B_values)
    
    experiment.run()
    experiment.plot_results()
    # experiment.display_results()
    # experiment.plot_q_trends()
    experiment.plot_q_by_round()
    experiment.plot_q_by_type()
    experiment.plot_all_savings()
    experiment.plot_all_accuracies()