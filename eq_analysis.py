# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Seed the random number generator
rng = np.random.default_rng(42)

# Determine the number of paths and points per path
points = 1000
paths = 50

# Create the initial set of random normal draws
mu, sigma = 0.0, 1.0
Z = rng.normal(mu, sigma, (paths, points))

# Define the time step size and t-axis
interval = [0.0, 1.0]
dt = (interval[1] - interval[0]) / (points - 1)
t_axis = np.linspace(interval[0], interval[1], points)

# Use Equation 3.2 from [Glasserman, 2003] to sample 50 standard brownian motion paths
W = np.zeros((paths, points))
for idx in range(points - 1):
    real_idx = idx + 1
    W[:, real_idx] = W[:, real_idx - 1] + np.sqrt(dt) * Z[:, idx]

# Plot these paths
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
    ax.plot(t_axis, W[path, :])
ax.set_title("Standard Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.show()

# Obtain the set of final path values
final_values = pd.DataFrame({'final_values': W[:, -1]})

# Estimate and plot the distribution of these final values with Seaborn
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.kdeplot(data=final_values, x='final_values', fill=True, ax=ax)
ax.set_title("Kernel Density Estimate of asset path final value distribution")
ax.set_ylim(0.0, 0.325)
ax.set_xlabel('Final Values of Asset Paths')
plt.show()

# Output the mean and stdev of these final values
print(final_values.mean(), final_values.std())

# Create a non-zero mean and non-unit standard deviation
mu_c, sigma_c = 5.0, 2.0

# Use Equation 3.3 from [Glasserman, 2003] to sample 50 brownian motion paths
X = np.zeros((paths, points))
for idx in range(points - 1):
    real_idx = idx + 1
    X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + sigma_c * np.sqrt(dt) * Z[:, idx]

# Plot these paths
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
    ax.plot(t_axis, X[path, :])
ax.set_title("Constant mean and standard deviation Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.show()