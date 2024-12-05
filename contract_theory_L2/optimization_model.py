import cvxpy as cp
import numpy as np

class OptimizationContractTheory:
    def __init__(self, I, T, N, sigma, eta, p, theta, q_max):
        """
            Handles the optimization logic for a single cluster.
        """
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