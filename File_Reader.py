import csv
import random
import numpy as np
def main():
    # Test the functions
    # then generate random data with 50 points
    data = generate_sine_wave_points()#[[random.uniform(-200, 200), random.gauss(-1000,1000 )] for _ in range(1000)]
    write_csv(data, "data.csv")
    data = read_xy_pairs_from_csv("data.csv")
    print(data)

def write_csv(data, filename):
    """Write data to a CSV file."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    file.close()
def read_xy_pairs_from_csv(filename):
    #intialize the list with given size
    pairs = []
    # Open the file in read mode
    with open(filename, "r") as file:
        # Create a CSV reader
        reader = csv.reader(file)
        # Read each row in the CSV file
        for row in reader:
            # Convert each element to a float and add to the list
            pairs.append([float(row[0]), float(row[1])])
    return pairs

def generate_weighted_random_numbers(size=100, preferred_range=(40, 60), outside_range=(0, 100), preferred_weight=0.7):
    # Separate the range into two parts: preferred and non-preferred
    preferred_numbers = np.random.randint(preferred_range[0], preferred_range[1], int(size * preferred_weight))
    outside_numbers = np.random.randint(outside_range[0], outside_range[1], int(size * (1 - preferred_weight)))

    # Concatenate the two lists and shuffle
    result = np.concatenate((preferred_numbers, outside_numbers))
    np.random.shuffle(result)
    
    return result.tolist()
def generate_sine_wave_points(num_points=200, x_range=(-100, 100), amplitude=500, frequency=1):
    # Generate x values evenly spaced within the given range
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    # Calculate y values as a sine wave based on x values, amplitude, and frequency
    y_values = amplitude * np.sin(frequency * x_values)
    # Combine x and y into pairs
    points = [[x, y] for x, y in zip(x_values, y_values)]
    return points
if __name__ == "__main__":
    main()
'''
1,2
2,10
4,40
7,60
8,90
15,20
19,-20
20,-70
22,-100
25,-140
27,-10
30,40
40,130
50,150
60,200
70,270
80,-100
90,-100
100,-150
110,0
120,10
200,130
300,244
400,-52
500,-100
'''