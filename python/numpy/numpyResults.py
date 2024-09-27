import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Task 1: Create a 2D array representing a 5x5 grid with random integers
grid_5x5 = np.random.randint(1, 100, size=(5, 5))
print("Task 1: 5x5 grid with random integers\n", grid_5x5)

# Task 2: Reshape a 1D array of size 9 into a 3x3 matrix
array_1d = np.arange(9)
matrix_3x3 = array_1d.reshape((3, 3))
print("\nTask 2: 3x3 matrix\n", matrix_3x3)

# Task 3: Create a 3D array and access a specific element
array_3d = np.random.randint(1, 100, size=(3, 3, 3))
print("\nTask 3: Element in 3D array\n", array_3d[1, 2, 0])

# Task 4: Apply mathematical functions on an array
array = np.arange(10)
sqrt_array = np.sqrt(array)
exp_array = np.exp(array)
sin_array = np.sin(array)
print("\nTask 4: Mathematical operations\nSquare Root:", sqrt_array)
print("Exponential:", exp_array)
print("Sine:", sin_array)

# Task 5: Compute statistics (mean, median, std)
random_data = np.random.rand(100)
mean_value = np.mean(random_data)
median_value = np.median(random_data)
std_value = np.std(random_data)
print("\nTask 5: Mean:", mean_value, "Median:", median_value, "Std:", std_value)

# Task 6: Element-wise operations between arrays
array_a = np.array([1, 2, 3])
array_b = np.array([4, 5, 6])
sum_array = array_a + array_b
product_array = array_a * array_b
print("\nTask 6: Element-wise sum and product\nSum:", sum_array, "Product:", product_array)

# Task 7: Broadcasting example
matrix = np.ones((3, 3))
vector = np.array([1, 2, 3])
broadcast_result = matrix + vector
print("\nTask 7: Broadcasting result\n", broadcast_result)

# Task 8: Multiply matrices using broadcasting
matrix_4x1 = np.random.rand(4, 1)
matrix_1x4 = np.random.rand(1, 4)
broadcast_mult = matrix_4x1 * matrix_1x4
print("\nTask 8: Broadcasting multiplication result\n", broadcast_mult)

# Task 9: Subtract the mean from each column
matrix = np.random.rand(4, 3)
mean_subtracted = matrix - matrix.mean(axis=0)
print("\nTask 9: Mean-subtracted matrix\n", mean_subtracted)

# Task 10: Matrix multiplication
matrix_a = np.random.rand(3, 3)
matrix_b = np.random.rand(3, 3)
matrix_product = np.dot(matrix_a, matrix_b)
print("\nTask 10: Matrix product\n", matrix_product)

# Task 11: Determinant and inverse of a matrix
matrix_c = np.random.rand(3, 3)
det = np.linalg.det(matrix_c)
inv = np.linalg.inv(matrix_c)
print("\nTask 11: Determinant:", det, "Inverse:\n", inv)

# Task 12: Solve a system of linear equations
coefficients = np.array([[2, -1], [-1, 2]])
constants = np.array([1, 0])
solution = np.linalg.solve(coefficients, constants)
print("\nTask 12: Solution to linear equations\n", solution)

# Task 13: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix_c)
print("\nTask 13: Eigenvalues\n", eigenvalues)
print("Eigenvectors\n", eigenvectors)

# Task 14: Generate random integers
random_integers = np.random.randint(50, 100, size=10)
print("\nTask 14: Random integers\n", random_integers)

# Task 15: Simulate rolling a dice
dice_rolls = np.random.randint(1, 7, size=100)
print("\nTask 15: Dice rolls simulation\n", dice_rolls)

# Task 16: Create random numbers from a normal distribution and visualize
normal_data = np.random.randn(1000)
plt.hist(normal_data, bins=30)
plt.title("Task 16: Histogram of normal distribution")
plt.show()

# Task 17: Shuffle an array and take a random sample
array = np.arange(10)
np.random.shuffle(array)
sample = np.random.choice(array, size=5, replace=False)
print("\nTask 17: Shuffled array:", array, "Random sample:", sample)

# Task 18: Create a Pandas DataFrame using NumPy array
df = pd.DataFrame(np.random.rand(5, 3), columns=["A", "B", "C"])
print("\nTask 18: Pandas DataFrame\n", df)

# Task 19: Plot using NumPy and Matplotlib
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Task 19: Sine Wave Plot")
plt.show()

# Task 20: Import a CSV into Pandas, convert a column to NumPy array
# df = pd.read_csv('data.csv')  # Example with real data
# column_as_array = df['column_name'].values

# Task 21: Compare Python loop vs NumPy sum
python_list = list(range(1000000))
start_time = time.time()
python_sum = sum(python_list)
python_time = time.time() - start_time

numpy_array = np.array(python_list)
start_time = time.time()
numpy_sum = np.sum(numpy_array)
numpy_time = time.time() - start_time

print(f"\nTask 21: Python sum time: {python_time:.6f}s, NumPy sum time: {numpy_time:.6f}s")

# Task 22: Vectorized vs loop operations
array = np.arange(1000000)
start_time = time.time()
loop_result = [i ** 2 for i in array]
loop_time = time.time() - start_time

start_time = time.time()
vectorized_result = array ** 2
vectorized_time = time.time() - start_time

print(f"\nTask 22: Loop time: {loop_time:.6f}s, Vectorized time: {vectorized_time:.6f}s")

# Task 23: Process weather data (example task with real dataset)
# weather_data = pd.read_csv('weather.csv')  # Example with real data
# weather_np = weather_data[['Temperature', 'Humidity']].to_numpy()
# mean_temp = np.mean(weather_np[:, 0])

# Task 24: Filter an array based on a threshold
array = np.random.rand(10)
filtered_array = array[array > 0.5]
print("\nTask 24: Filtered array\n", filtered_array)

# Task 25: Group data based on a condition
grouped_data = array[array > array.mean()]
print("\nTask 25: Grouped data based on mean\n", grouped_data)

# Task 26: Normalize dataset (features)
dataset = np.random.rand(100, 5)
normalized_data = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
print("\nTask 26: Normalized data\n", normalized_data)

# Task 27: Split dataset into training and testing sets
train_data = dataset[:80]
test_data = dataset[80:]
print("\nTask 27: Training and Testing sets\nTraining data size:", train_data.shape, "Test data size:", test_data.shape)

# Task 28: Linear regression (manual implementation using NumPy)
x = np.random.rand(100)
y = 3 * x + np.random.randn(100) * 0.1  # y = 3x + noise
A = np.vstack([x, np.ones_like(x)]).T
slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
print("\nTask 28: Linear regression slope:", slope, "Intercept:", intercept)

# Task 29: Simulate projectile motion
time_points = np.linspace(0, 10, num=100)
initial_velocity = 10
g = 9.81
heights = initial_velocity * time_points - 0.5 * g * time_points ** 2
plt.plot(time_points, heights)
plt.title("Task 29: Projectile motion")
plt.show()

# Task 30: Solve differential equation (simple ODE)
# Example placeholder for differential equation simulation

# Task 31: Fourier transform of a sine wave signal
signal = np.sin(2 * np.pi * 5 * time_points)
fourier_transform = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=time_points[1] - time_points[0])
plt.plot(frequencies, np.abs(fourier_transform))
plt.title("Task 31: Fourier Transform")
plt.show()

# Task 32: Load and manipulate an image as NumPy array
# image = plt.imread('image.png')  # Example with real image
# inverted_image = 255 - image

# Task 33: Simulate coin tosses and calculate probability
coin_tosses = np.random.choice(['Heads', 'Tails'], size=1000)
heads_probability = np.sum(coin_tosses == 'Heads') / 1000
print("\nTask 33: Heads probability\n", heads_probability)

