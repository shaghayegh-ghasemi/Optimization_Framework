# Optimization Contract Theory Framework

This repository implements an optimization framework to solve budget-constrained problems. The framework determines optimal data contributions (\( q \)) for user types across multiple rounds while satisfying budget constraints and maximizing the server's utility.

## Features

- **Flexible Configuration**: Easily set parameters for user types, computation costs, rounds, and budget.
- **Convex Optimization**: Leverages the CVXPY library to solve constrained optimization problems.
- **Dynamic Budget Allocation**: Computes budget allocation across rounds based on reward and contribution dynamics.
- **Utility Maximization**: Optimizes server utility while ensuring constraints like budget limits and user contributions.
- **Plotting Relationships**: Analyze and visualize the relationship between budget (\( B \)) and contributions (\( q \)).

---

### Prerequisites

- Python 3.7+
- CVXPY (for solving optimization problems)
- NumPy (for numerical operations)
- Matplotlib (for plotting relationships)

## Installation

To set up the project environment and install the necessary dependencies, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/shaghayegh5ghasemi/Optimization_Framework.git
cd Optimization_Framework
```

### 2. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 3. Running the Project

Once the dependencies are installed, you can run the scripts or use the framework according to the instructions in the code.
