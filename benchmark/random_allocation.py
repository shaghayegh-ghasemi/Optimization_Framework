import numpy as np

class RandomAllocationBaseline:
    def __init__(self, total_budget, num_servers, num_users_per_server):
        self.total_budget = total_budget
        self.num_servers = num_servers
        self.num_users_per_server = num_users_per_server

    def allocate_budget(self):
        # Randomly distribute budget among local servers
        server_budgets = np.random.dirichlet(np.ones(self.num_servers)) * self.total_budget
        
        user_budgets = []
        for budget in server_budgets:
            user_alloc = np.random.dirichlet(np.ones(self.num_users_per_server)) * budget
            user_budgets.append(user_alloc)

        allocation = {
            "server_budgets": server_budgets,
            "user_budgets": user_budgets
        }
        return allocation

    def run(self):
        allocation = self.allocate_budget()
        print("Random Allocation Baseline Results:")
        print(f"Server budgets: {allocation['server_budgets']}")
        return allocation

    