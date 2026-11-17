import numpy as np
import matplotlib.pyplot as plt

# Define target function and nodes
def f(x):
    return np.sqrt(x + 2)

# Interpolation nodes
nodes_P2 = np.array([0, 1, 3])
y_P2 = f(nodes_P2)

nodes_P3 = np.array([0, 1, 3, 2])
y_P3 = f(nodes_P3)

# 1. Evaluate f(x) at initial nodes
print("1. Function values at nodes:")
for x_val in nodes_P2:
    print(f"   f({x_val}) = {f(x_val):.6f}")

# 2. Lagrange Interpolation function for P2
def P2(x):
    L0 = (x - 1)*(x - 3) / ((0 - 1)*(0 - 3))
    L1 = (x - 0)*(x - 3) / ((1 - 0)*(1 - 3))
    L2 = (x - 0)*(x - 1) / ((3 - 0)*(3 - 1))
    return f(0)*L0 + f(1)*L1 + f(3)*L2

# 3. Evaluate at test points and compute errors
test_points = np.array([0.5, 2.0, 3.5])
f_test = f(test_points)
P2_test = P2(test_points)
errors_P2 = np.abs(f_test - P2_test)

print("\n3. Evaluation at test points:")
for tp, ft, pt, err in zip(test_points, f_test, P2_test, errors_P2):
    print(f"   x = {tp:.1f} | f(x) = {ft:.6f} | P2(x) = {pt:.6f} | error = {err:.6f}")

# 4. Error norms
L5_norm = (np.sum(errors_P2**5))**(1/5)
max_sum_norm = np.max(errors_P2) + np.sum(errors_P2)

print("\n4. Error Norms:")
print(f"   Discrete L5 norm: {L5_norm:.6f}")
print(f"   Max+Sum norm:     {max_sum_norm:.6f}")

# 5. Polynomial P3 using Lagrange / Polyfit
poly_P3_coeffs = np.polyfit(nodes_P3, y_P3, 3)
P3 = np.poly1d(poly_P3_coeffs)

print("\n5. Cubic Polynomial P3(x) coefficients (highest degree first):")
print(f"   {poly_P3_coeffs}")

# 6 & 7. Plotting
x_dense = np.linspace(0, 4, 500)
f_dense = f(x_dense)
P2_dense = P2(x_dense)
P3_dense = P3(x_dense)

err_P2_dense = np.abs(f_dense - P2_dense)
err_P3_dense = np.abs(f_dense - P3_dense)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Functions
ax1.plot(x_dense, f_dense, 'k-', label=r'$f(x) = \sqrt{x+2}$', linewidth=2)
ax1.plot(x_dense, P2_dense, 'r--', label=r'$P_2(x)$ (Quadratic)')
ax1.plot(x_dense, P3_dense, 'b-.', label=r'$P_3(x)$ (Cubic)')
ax1.scatter(nodes_P2, y_P2, color='red', zorder=5, label='Nodes for P2')
ax1.scatter([2], [f(2)], color='blue', zorder=5, label='Added Node x3=2')
ax1.set_title('Comparison of $f(x)$, $P_2(x)$, and $P_3(x)$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# Plot 2: Errors
ax2.plot(x_dense, err_P2_dense, 'r-', label=r'$|f(x) - P_2(x)|$')
ax2.plot(x_dense, err_P3_dense, 'b-', label=r'$|f(x) - P_3(x)|$')
ax2.set_title('Pointwise Error Distribution over [0, 4]')
ax2.set_xlabel('x')
ax2.set_ylabel('Absolute Error')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()
