
# Advanced Test Answers on Pandas, Matplotlib, Seaborn, NumPy, Plotly, and Python Functions

# Section 1: Pandas

# 1. Handling missing data
import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4]
}
df = pd.DataFrame(data)

# Method 1: Drop missing values
df_dropped = df.dropna()

# Method 2: Fill missing values with the mean
df_filled = df.fillna(df.mean())

# 2. Find top 3 rows with highest values in a specific column
top_rows = df.nlargest(3, 'A')

# 3. Using groupby to calculate sum
grouped_sum = df.groupby(['A']).sum()

# 4. Function to normalize a column
def normalize_column(df, col_name):
    df[col_name] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    return df

# 5. Merging two DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key', how='inner')


# Section 2: Matplotlib

# 6. Customizing a Matplotlib plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot([1, 2, 3], [4, 5, 6], color='blue', linestyle='--', marker='o', markersize=10)
plt.title('Customized Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

# 7. Creating a subplot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot([1, 2, 3], [4, 5, 6], label='Line')
axs[1].bar([1, 2, 3], [4, 5, 6], label='Bar')
axs[2].scatter([1, 2, 3], [4, 5, 6], label='Scatter')
for ax in axs:
    ax.legend()
plt.show()

# 8. Adding a secondary y-axis
fig, ax1 = plt.subplots()
ax1.plot([1, 2, 3], [4, 5, 6], color='blue')
ax1.set_ylabel('Primary Y-axis', color='blue')

ax2 = ax1.twinx()
ax2.plot([1, 2, 3], [10, 20, 30], color='red')
ax2.set_ylabel('Secondary Y-axis', color='red')
plt.show()


# Section 3: Seaborn

# 9. Differences between sns.barplot() and sns.countplot()
# sns.barplot() is used for showing mean values of a categorical variable,
# while sns.countplot() is used for showing counts of observations in each categorical bin.

import seaborn as sns

# Example for sns.barplot()
sns.barplot(x='A', y='B', data=df)

# Example for sns.countplot()
sns.countplot(x='A', data=df)

# 10. Generating a violin plot
sns.violinplot(x='A', y='B', data=df)
plt.show()

# Insights: The violin plot shows the distribution of the data and where the data points are concentrated.

# 11. Improving readability of Seaborn visualizations
sns.set(style='whitegrid')  # Set background style
sns.barplot(x='A', y='B', data=df)
plt.title('Improved Bar Plot')
plt.show()


# Section 4: NumPy

# 12. Function to return the row index of the maximum value in each row
def max_row_index(arr):
    return np.argmax(arr, axis=1)

array_2d = np.array([[1, 2, 3], [4, 5, 6]])
max_indices = max_row_index(array_2d)

# 13. Element-wise operations between two NumPy arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
sum_array = array1 + array2
product_array = array1 * array2

# 14. Creating a 3D array and computing the sum along each axis
array_3d = np.random.rand(2, 3, 4)
sum_along_axis_0 = np.sum(array_3d, axis=0)
sum_along_axis_1 = np.sum(array_3d, axis=1)
sum_along_axis_2 = np.sum(array_3d, axis=2)


# Section 5: Plotly

# 15. Creating interactive plots using Plotly
import plotly.express as px

df_plotly = px.data.iris()
fig = px.scatter(df_plotly, x='sepal_width', y='sepal_length', color='species', hover_data=['petal_length'])
fig.show()

# 16. Creating a 3D surface plot from a mathematical function
import numpy as np
import plotly.graph_objects as go

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
fig.show()


# Section 6: Advanced Python Functions, Loops, and Map

# 17. Function to filter dictionaries by a specific key
def filter_dicts(dicts, key, threshold):
    return [d for d in dicts if d.get(key, 0) > threshold]

dicts = [{'a': 1}, {'a': 2}, {'a': 3}]
filtered_dicts = filter_dicts(dicts, 'a', 1)

# 18. Using list comprehensions to create a new list
numbers = [1, 2, 3, 4]
squared_numbers = [x**2 for x in numbers]

# 19. Recursive function to calculate factorial
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

# 20. Using filter() function to return even numbers
numbers_list = [1, 2, 3, 4, 5, 6]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers_list))
