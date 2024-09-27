import numpy as np
import pandas as pd

# print(toyPrices - 2) numpy integrated loop
toyPrices = np.array([5,8,3,6])
print(toyPrices - 2)

# print(toyPrices - 2) -- Python list Not possible. Causes an error
toyPricess = [5,8,3,6]
for i in range(len(toyPricess)):
    toyPricess[i] -= 2
print(toyPricess)


# Create a Series using a NumPy array of ages but customize the indices to be the names that correspond to each age
ages = np.array([13,25,19])
series1 = pd.Series(ages,index=['Emma', 'Swetha', 'Serajh'])
print(series1)


# Index is first the default 0 till n, after set_index it changes to name
dataf = pd.DataFrame([
    ['John Smith','123 Main St',34],
    ['Jane Doe', '456 Maple Ave',28],
    ['Joe Schmo', '789 Broadway',51]
    ],
    columns=['name','address','age'])

dataf.set_index('name')

### 1. **Array Creation**
print("### 1. Array Creation")

# Create an array with specific values
arr = np.array([1, 2, 3, 4, 5])
print("Array with specific values:", arr)

# Generate an array of zeros
arr_zeros = np.zeros(5)
print("Array of zeros:", arr_zeros)

# Generate an array of ones
arr_ones = np.ones(5)
print("Array of ones:", arr_ones)

# Create an empty array and fill it with random values
arr_empty = np.empty(5)
arr_empty[:] = np.random.rand(5)
print("Empty array filled with random values:", arr_empty)

# Create an array with a range of values from 1 to 10
arr_range = np.arange(1, 11)
print("Array with a range of values:", arr_range)

# Create an array with evenly spaced values from 0 to 10
arr_linspace = np.linspace(0, 10, 5)
print("Array with evenly spaced values:", arr_linspace)

# Generate an identity matrix
arr_eye = np.eye(3)
print("Identity matrix:", arr_eye)

# Create an array of random numbers
arr_rand = np.random.rand(5)
print("Array of random numbers (uniform distribution):", arr_rand)

arr_randn = np.random.randn(5)
print("Array of random numbers (standard normal distribution):", arr_randn)

### 2. **Array Manipulation**
print("\n### 2. Array Manipulation")

# Reshape an array from 1D to 2D
arr_1d = np.array([1, 2, 3, 4, 5, 6])
arr_2d = arr_1d.reshape(2, 3)
print("Reshaped array from 1D to 2D:", arr_2d)

# Transpose an array
arr_transpose = arr_2d.transpose()
print("Transposed array:", arr_transpose)

# Flatten a 2D array into a 1D array
arr_flatten = arr_2d.ravel()
print("Flattened array:", arr_flatten)

# Stack two arrays vertically
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr_stacked = np.vstack((arr1, arr2))
print("Stacked arrays vertically:", arr_stacked)

# Stack two arrays horizontally
arr_hstacked = np.hstack((arr1, arr2))
print("Stacked arrays horizontally:", arr_hstacked)

# Concatenate arrays along an axis
arr_concat = np.concatenate((arr1, arr2), axis=0)
print("Concatenated arrays along axis 0:", arr_concat)

# Split an array into multiple sub-arrays
arr_split = np.split(arr_concat, 2)
print("Split array into multiple sub-arrays:", arr_split)

# Repeat an array
arr_repeated = np.tile(arr1, 2)
print("Repeated array:", arr_repeated)

# Reverse the elements of an array
arr_reversed = np.flip(arr1)
print("Reversed array:", arr_reversed)

### 3. **Mathematical Operations**
print("\n### 3. Mathematical Operations")

# Element-wise addition
arr_add = np.add(arr1, arr2)
print("Element-wise addition:", arr_add)

# Element-wise subtraction
arr_subtract = np.subtract(arr1, arr2)
print("Element-wise subtraction:", arr_subtract)

# Element-wise multiplication
arr_multiply = np.multiply(arr1, arr2)
print("Element-wise multiplication:", arr_multiply)

# Element-wise division
arr_divide = np.divide(arr1, arr2)
print("Element-wise division:", arr_divide)

# Element-wise exponentiation
arr_power = np.power(arr1, 2)
print("Element-wise exponentiation:", arr_power)

# Dot product of two arrays
arr_dot = np.dot(arr1, arr2)
print("Dot product of two arrays:", arr_dot)

# Matrix multiplication
arr_matmul = np.matmul(arr1.reshape(1, -1), arr2.reshape(-1, 1))
print("Matrix multiplication:", arr_matmul)

# Compute the inverse of a matrix
arr_inv = np.linalg.inv(np.array([[1, 2], [3, 4]]))
print("Inverse of a matrix:", arr_inv)

# Compute the eigenvalues and eigenvectors of a matrix
arr_eig = np.linalg.eig(np.array([[1, 2], [3, 4]]))
print("Eigenvalues and eigenvectors of a matrix:", arr_eig)

# Compute the determinant of a matrix
arr_det = np.linalg.det(np.array([[1, 2], [3, 4]]))
print("Determinant of a matrix:", arr_det)

### 4. **Statistical Functions**
print("\n### 4. Statistical Functions")

# Compute the mean of an array
arr_mean = np.mean(arr1)
print("Mean of an array:", arr_mean)

# Compute the median of an array
arr_median = np.median(arr1)
print("Median of an array:", arr_median)

# Compute the variance of an array
arr_var = np.var(arr1)
print("Variance of an array:", arr_var)

# Compute the standard deviation of an array
arr_std = np.std(arr1)
print("Standard deviation of an array:", arr_std)

# Compute the sum of an array
arr_sum = np.sum(arr1)
print("Sum of an array:", arr_sum)

# Compute the minimum value of an array
arr_min = np.min(arr1)
print("Minimum value of an array:", arr_min)

# Compute the maximum value of an array
arr_max = np.max(arr1)
print("Maximum value of an array:", arr_max)

# Compute the nth percentile of an array
arr_percentile = np.percentile(arr1, 50)
print("50th percentile of an array:", arr_percentile)

### 5. **Random Number Generation**
print("\n### 5. Random Number Generation")

# Set the seed for random number generation
np.random.seed(0)

# Generate a random sample from a given array
arr_random_sample = np.random.choice(arr1, 2)
print("Random sample from an array:", arr_random_sample)

# Generate random numbers from a uniform distribution
arr_uniform = np.random.uniform(0, 1, 5)
print("Random numbers from a uniform distribution:", arr_uniform)

# Generate random numbers from a normal distribution
arr_normal = np.random.normal(0, 1, 5)
print("Random numbers from a normal distribution:", arr_normal)

### 6. **Linear Algebra Functions**
print("\n### 6. Linear Algebra Functions")

# Compute the inverse of a matrix
arr_inv = np.linalg.inv(np.array([[1, 2], [3, 4]]))
print("Inverse of a matrix:", arr_inv)

# Compute the eigenvalues and eigenvectors of a matrix
arr_eig = np.linalg.eig(np.array([[1, 2], [3, 4]]))
print("Eigenvalues and eigenv")