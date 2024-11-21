import numpy as np
from mpmath import mp, cos, sin, tan, cot, sec, csc, atan2 # use when need extreme high precision
import matplotlib.pyplot as plt 
import math
import MatrixOperator as mo
import random
from scipy.misc import derivative
from scipy.integrate import quad
import time 
import File_Reader as fr


def main():
    """Main function to execute the script."""
    #start counting time
    start_time = time.time()
    mp.dps = 200 # Set the precision to 300 decimal places
    # Plot the function
    points = fr.read_xy_pairs_from_csv("data.csv")
    plot_points(points)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    plt.show() 
    #end counting time
#//////////////////////////////////////////////////////////////////////////////////////////
# Function Definitions
def f1(x, N=0):
    return x*2
def p1(theta):
    return np.sin(theta**2)
def implitcit_cartesian_e1(x, y, k):
    return np.sqrt(x**2 + y**2) - np.sin( (np.arctan2(y, x) +2*k*np.pi )**2)
def implitcit_polar_e1(r, theta):
    return
#//////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////
#HELPER FUNCTIONS
def cos_v(x): # use when need extreme high precision
    result = np.vectorize(cos)(x).astype(float)
    return result
def sin_v(x):# use when need extreme high precision
    result = np.vectorize(sin)(x).astype(float)
    return result
def tan_v(x):# use when need extreme high precision
    result = np.vectorize(tan)(x).astype(float)
    return result
def cot_v(x):# use when need extreme high precision 
    result = np.vectorize(cot)(x).astype(float)
    return result

def sec_v(x):# use when need extreme high precision
    result = np.vectorize(sec)(x).astype(float)
    return result

def csc_v(x):# use when need extreme high precision
    result = np.vectorize(csc)(x).astype(float)
    return result

def atan2_v(y, x):# use when need extreme high precision
    result = np.vectorize(atan2)(y, x).astype(float)
    return result

def Power_series(x, max_n):
    result = np.zeros_like(x, dtype=float)  # Initialize result to zeros with object dtype
    result += 1 
    n = 1  # Initialize n 
    while n <= max_n:
        #print(f"f_{n}: {factorial(n)}") 
        result += n
        n += 1
    return result
def Reduction_integral(x, n):
    result = np.zeros_like(x, dtype=float)  # Initialize result to zeros with object dtype
    if n == 0:
        return x
    elif n == 1:
        return -np.cos(x)
    else:
        result += ( (-sin_v(x)**(n-1) * cos_v(x))/(n)) + (((n-1)/n)*Reduction_integral(x, n-2))
    return result

#//////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////
# PLotting functions
def Plot_catersian_function(function: callable, X_start: int = -10, X_end:int = 20, Y_start: int = -50, Y_end:int = 50, Coef: int = 0): #N is optional and for power series
    x_values = np.linspace(X_start, X_end, 10000)  # Generate x values (increased for better resolution)
    y_values = function(x_values, Coef)  # Calculate y values
    plt.plot(x_values, y_values, label=f" (F(x))", color = (random.random(), random.random(), random.random())) #random color
    #default setting
    plt.title(f"F(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=1)  # X-axis
    plt.axvline(0, color='black', linewidth=1)  # Y-axis
    plt.grid(True)
    plt.legend()
    plt.xlim(X_start, X_end)       # Set x-axis limits
    plt.ylim(Y_start, Y_end)     # Set y-axis limits (adjust as needed)
    plt.show()


def Plot_polar_function(function: callable, Theta_start: int = 0, Theta_end:int = 2*np.pi, Num_points: int = 1000):
    # Define the angle range
    theta = np.linspace(Theta_start, Theta_end, Num_points)
    # Define the polar equation, e.g., r = 1 + 0.5 * sin(theta)
    r = function(theta)
    # Plot the polar equation
    ax = plt.subplot(projection='polar')
    ax.plot(theta, r)
    ax.set_title("Polar Plot")
    #default setting
    plt.title(f"P(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=1)  # X-axis
    plt.axvline(0, color='black', linewidth=1)  # Y-axis
    plt.grid(True)
    plt.figure()
       
def Plot_parametric_of_cartesian_form():
    # Define the parameter range
    t = np.linspace(-20, 20, 800)
    # Define the parametric equations
    x = t**2
    y = t
    # Plot the parametric equations with axis labels
    plt.plot(x, y, label="Parametric Plot")
    #default setting
    plt.title(f"F(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=1)  # X-axis
    plt.axvline(0, color='black', linewidth=1)  # Y-axis
    plt.grid(True)
    plt.legend()
    plt.show()

def Plot_parametric_of_polar_form():
    # Define the parameter range
    t = np.linspace(-20, 20, 800)
    # get theta from t
    theta = np.arctan2( t**2,t) #y/x
    # get max and min of theta
    theta_min = min(theta)
    theta_max = max(theta)
    print(theta)
    print(f"theta_min: {theta_min}, theta_max: {theta_max}")
    # get r from t
    r = np.sqrt((t**2)**2 + (t)**2)

    # Define the parametric equations
    # Plot the polar_parametric equations with axis labels
    # Plot the polar equation
    ax = plt.subplot(projection='polar')
    ax.plot(theta, r)
    ax.set_title("Polar-Parametric Plot")
    plt.title(f"F(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=1)  # X-axis
    plt.axvline(0, color='black', linewidth=1)  # Y-axis
    plt.grid(True)
    plt.show()

def plot_implicit_cartesian_equation(implicit_func, xlim=(-15, 15), ylim=(-15, 15), num_points=500,k=0):
    """
    Plots the implicit function defined by F(x, y) = 0.

    Parameters:
    - implicit_func: A function F(x, y) that defines the implicit equation.
    - xlim: Tuple (xmin, xmax) defining the limits of the x-axis.
    - ylim: Tuple (ymin, ymax) defining the limits of the y-axis.
    - num_points: Number of points to use in each direction for the grid.
    """
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], num_points)
    y = np.linspace(ylim[0], ylim[1], num_points)
    X, Y = np.meshgrid(x, y)

    # Calculate F(X, Y)

    Z = implicit_func(X, Y,k)

    # Plot the implicit function
    #plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, levels=[0], colors='blue')  # Plot the contour where F(X, Y) = 0
    plt.title(f'Implicit Plot of the Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.axis('equal')  # Equal scaling for x and y axes

def Plot_implicit_polar_equation(implicit_func, theta_range=(0, 2 * np.pi), r_range=(0, 2), num_points=400):
    """
    Plots the implicit polar function defined by F(r, theta) = 0.

    Parameters:
    - implicit_func: A function F(r, theta) that defines the implicit equation.
    - theta_range: Tuple (theta_min, theta_max) defining the limits for theta.
    - r_range: Tuple (r_min, r_max) defining the limits for r.
    - num_points: Number of points to use in each direction for the grid.
    """
    # Create a grid of theta and r values
    theta = np.linspace(theta_range[0], theta_range[1], num_points)
    r = np.linspace(r_range[0], r_range[1], num_points)
    R, Theta = np.meshgrid(r, theta)

    # Calculate the implicit function
    Z = implicit_func(R, Theta)

    # Convert polar to Cartesian coordinates
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Plot the implicit function
    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, levels=[0], colors='blue')  # Plot the contour where F(R, Theta) = 0
    plt.title('Implicit Polar Plot of the Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.show()


def plot_points(points: list[list[int]], color="blue", marker="o", title="Scatter Plot of Points"):
    """
    Plots an array of (x, y) points.

    Parameters:
    - points (list of lists): List of [x, y] pairs to plot.
    - color (str): Color of the points (default is "blue").
    - marker (str): Marker style for points (default is "o").
    - title (str): Title of the plot.
    """
    # Separate x and y values
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    # Plot the points
    plt.scatter(x_values, y_values, color=color, marker=marker, label="Points")

    # Add labels and title
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title(title)
    #draw axis
    plt.axhline(0, color='black', linewidth=1)  # X-axis
    plt.axvline(0, color='black', linewidth=1)  # Y-axis
    # Show grid and legend
    plt.grid(True)
    plt.legend()
    #set view limit
    #plt.xlim(-20, 20)
    #plt.ylim(-20, 20)
    # Display the plot






def Derivative(x: float) -> float:
    return derivative(function, x, dx=1e-6)

def Integral(x: float) -> float: 
    return quad(function, 0, x)[0]

def Repeated_Multiplication(max_m: int, n: int) -> int:
    result = 1
    m = 0
    while m <= max_m:
        result *= (n-(3*m+2))
        m += 1
    return result
def polynomial_regression_plot(points, degree=2):
    # Separate points into x and y arrays
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Fit polynomial to data
    coefficients = np.polyfit(x, y, degree)

    # Create a polynomial function based on the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate x values for the regression line
    x_fit = np.linspace(min(x), max(x), 1000)

    # Generate y values for the regression line
    y_fit = polynomial(x_fit)

    # Plot original data points
    plt.scatter(x, y, color='blue', label='Data Points')

    # Plot the polynomial regression line
    plt.plot(x_fit, y_fit, color='red', label=f'Polynomial Fit (Degree {degree})')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()
    plt.grid()
    #axis
    plt.axhline(0, color='black', linewidth=1)  # X-axis
    plt.axvline(0, color='black', linewidth=1)  # Y-axis 
    # Show plot
    plt.show()

    # Return coefficients for further use if needed
    return coefficients
if __name__ == "__main__":
    main()  # Run the main function if executed as a script
