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

    def run(self):
        """
        Run the optimization problem for each B value and store the results.
        """
        for B in self.B_values:
            q, utility = self.opt_problem.solve(B)
            self.results.append((B, utility, q))

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

        Compute the available budget B_t for each round:
        B_t = α_t * (B - Σ_{k=1}^{t-1} Σ_{i=1}^I p_i * N * R_k^i)
        """
        cumulative_R = cp.sum(cp.multiply(self.p[:, None] * self.N, self.compute_reward()), axis=0)  # Shape (T,)

        B_t = []  # This will store the budget values for each round t

        # Calculate α_t (it should be a vector of length T)
        alpha_t = np.array([self.q_max[t] / np.sum(self.q_max[t:]) for t in range(self.T)])

        for t in range(self.T):
            if t == 0:
                B_t.append(alpha_t[t] * self.B)  # First round's budget
            else:
                # Use cumulative reward up to the current round
                B_t.append(alpha_t[t] * (self.B - cp.sum(cumulative_R[:t])))

        return cp.hstack(B_t)  # This should return a vector of shape (T,)

    # def compute_delta_t_i(self):
    #     pass

    def compute_server_utility(self, R):
        """
        Compute the utility U:
        U_t = Σ_{i=1}^I p_i * N * (σ * log(1 + η * q_i^t) - R_i^t)
        """
        utility = 0
        for t in range(self.T):
            for i in range(self.I):
                utility += (
                    self.p[i]
                    * self.N
                    * (self.sigma * cp.log(1 + self.eta * self.q[i, t]) - R[i, t])
                )
        return -1*utility

    def solve(self, B_value):
        """
        Define and solve the optimization problem:
        Minimize U subject to q <= q_max and budget constraints.
        """
        # Assign the budget value to the symbolic parameter
        self.B.value = B_value

        # Compute dependent variables
        R = self.compute_reward()
        B_t = self.compute_B_t()

        # Constraints
        constraints = [
            self.q[:, t] <= self.q_max[t] for t in range(self.T)
        ]  # Max contribution
        constraints += [
            cp.sum(cp.multiply(self.p[:, None] * self.N, R[:, t])) <= B_t[t]
            for t in range(self.T)
        ]  # Budget constraints
        constraints += [
            self.q >= 0
        ] # Positive data distributions

        # Objective
        utility = self.compute_server_utility(R)

        # Problem definition
        problem = cp.Problem(cp.Minimize(utility), constraints)
        problem.solve()

        return self.q.value, utility.value
        
        
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
    # experiment.plot_results()
    # experiment.display_results()
    # experiment.plot_q_trends()
    experiment.plot_q_by_round()
    experiment.plot_q_by_type()