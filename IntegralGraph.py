import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return 1/np.sin(x)**2 #1/np.sqrt(1-np.sin(x)**2)  # Example function

# Numerical integration using the trapezoidal rule
def numerical_integral(f, x_values):
    integral_values = np.zeros_like(x_values)  # Initialize array to store integral values
    for i in range(1, len(x_values)):
        # Trapezoidal rule: (f(x_i) + f(x_{i-1})) / 2 * (x_i - x_{i-1})
        integral_values[i] = integral_values[i-1] + 0.5 * (f(x_values[i]) + f(x_values[i-1])) * (x_values[i] - x_values[i-1])
    return integral_values

# Set up the x-values for plotting
x_values = np.linspace(-20, 20, 10000)  # Define the range of x values

# Compute the numerical integral
integral_values = numerical_integral(f, x_values)

# Plot the original function f(x) and its integral
plt.plot(x_values, f(x_values), label="f(x) = sin^2(x)", color='blue')
plt.plot(x_values, integral_values, label="Integral of f(x)", color='red')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Function and Its Numerical Integral")
plt.legend()
plt.grid(True)
plt.xlim(-10, 10)
plt.ylim(-5, 5)
plt.show()
