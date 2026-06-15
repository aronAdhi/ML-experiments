import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Forces the interactive window

import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Dataset
X = np.array([1.0, 2.0])
y = np.array([2.0, 4.0]) 

# 2. MSE function
def compute_mse(m, b, X, y):
    mse = 0
    for xi, yi in zip(X, y):
        prediction = m * xi + b
        mse += (prediction - yi) ** 2
    return mse / len(X)

# 3. Create grid
m_vals = np.linspace(-2, 6, 100)
b_vals = np.linspace(-4, 4, 100)
M, B = np.meshgrid(m_vals, b_vals)

Z = np.zeros(M.shape)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        Z[i, j] = compute_mse(M[i, j], B[i, j], X, y)

# 4. Plotting
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a bit more transparency (alpha=0.6) so points are visible
surface = ax.plot_surface(M, B, Z, cmap=cm.coolwarm, edgecolor='none', alpha=0.6)

# ----------------------------------------------------
# 5. Plotting Specific Points and Labels
# ----------------------------------------------------

# Point A: Perfect fit (m=2, b=0) -> Loss is 0
mA, bA = 2.0, 0.0
lossA = compute_mse(mA, bA, X, y)
ax.scatter(mA, bA, lossA, color='green', s=100, label='Perfect Fit A (m=2, b=0)', zorder=5)
ax.text(mA, bA, lossA + 2, " Perfect A (Loss=0)", color='green', fontweight='bold')

# Point B: Another Perfect fit (m=1, b=2) -> Loss is also 0!
mB, bB = 1.0, 2.0
lossB = compute_mse(mB, bB, X, y)
ax.scatter(mB, bB, lossB, color='forestgreen', s=100, label='Perfect Fit B (m=1, b=2)', zorder=5)
ax.text(mB, bB, lossB + 2, " Perfect B (Loss=0)", color='forestgreen', fontweight='bold')

# Point C: High Error Point (m=5, b=-2) -> High Loss
mC, bC = 5.0, -2.0
lossC = compute_mse(mC, bC, X, y)
ax.scatter(mC, bC, lossC, color='black', s=100, label='High Loss Point', zorder=5)
ax.text(mC, bC, lossC + 2, " High Loss", color='black', fontweight='bold')

# Draw a dashed line between A and B to show the "spine" or valley floor
ax.plot([mA, mB], [bA, bB], [lossA, lossB], color='green', linestyle='--', linewidth=2)
# ----------------------------------------------------

# Labels and styling
ax.set_title('Loss Surface with Specific Model Coordinates', fontsize=14, pad=20)
ax.set_xlabel('Slope (m)', fontsize=12)
ax.set_ylabel('Bias (b)', fontsize=12)
ax.set_zlabel('Mean Squared Error (Loss)', fontsize=12)

fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Loss Value')
ax.legend(loc='upper left')

print("Opening window... Look for the Green points at the bottom of the fold!")
plt.show()