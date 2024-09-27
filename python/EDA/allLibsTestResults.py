import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Section 1: Pandas

# 1. Handling missing data
def handle_missing_data(df):
    # Method 1: Drop rows with missing values
    df_dropped = df.dropna()
    
    # Method 2: Fill missing values
    df_filled = df.copy()
    for column in df_filled.columns:
        if df_filled[column].dtype.kind in 'biufc':  # Check if column is numeric
            df_filled[column].fillna(df_filled[column].mean(), inplace=True)
        else:
            df_filled[column].fillna(df_filled[column].mode()[0], inplace=True)
    
    return df_dropped, df_filled

# 2. Find top 3 rows with highest values in a column
def top_3_rows(df, column):
    return df.nlargest(3, column)

# 3. Group by two columns and calculate sum
def group_and_sum(df, group_cols, sum_col):
    return df.groupby(group_cols)[sum_col].sum()

# 4. Normalize column values
def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

# 5. Merge DataFrames
def merge_dataframes(df1, df2, key):
    return pd.merge(df1, df2, on=key, how='inner')

# Section 2: Matplotlib

# 6. Customize Matplotlib plot
def custom_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='red', linestyle='--', linewidth=2)
    plt.title('Customized Sine Wave', fontsize=16)
    plt.xlabel('X-axis', fontweight='bold')
    plt.ylabel('Y-axis', fontweight='bold')
    plt.grid(True, linestyle=':')
    plt.show()

# 7. Subplot with three different plots
def subplot_example():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.linspace(0, 10, 100)
    
    ax1.plot(x, np.sin(x))
    ax1.set_title('Line Plot')
    
    ax2.bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
    ax2.set_title('Bar Plot')
    
    ax3.scatter(np.random.rand(50), np.random.rand(50))
    ax3.set_title('Scatter Plot')
    
    plt.tight_layout()
    plt.show()

# 8. Secondary y-axis
def secondary_y_axis():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.exp(x)
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(x, y1, 'b-', label='Sine')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Sine', color='b')
    
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-', label='Exponential')
    ax2.set_ylabel('Exponential', color='r')
    
    plt.title('Sine and Exponential Functions')
    plt.show()

# Section 3: Seaborn

# 9. Barplot vs Countplot
# sns.barplot() is used when you have a numeric value for each category
# sns.countplot() is used when you want to count occurrences of each category

def barplot_example(df):
    sns.barplot(x='category', y='value', data=df)
    plt.title('Bar Plot: Average Value by Category')
    plt.show()

def countplot_example(df):
    sns.countplot(x='category', data=df)
    plt.title('Count Plot: Frequency of Categories')
    plt.show()

# 10. Violin plot
def violin_plot(df):
    sns.violinplot(x='category', y='value', data=df)
    plt.title('Violin Plot: Distribution of Values by Category')
    plt.show()

# 11. Improve readability in Seaborn
def improved_seaborn_plot(df):
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='category', y='value', data=df)
    
    plt.title('Improved Box Plot', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Section 4: NumPy

# 12. Row index of maximum value in each row
def max_row_index(arr):
    return np.argmax(arr, axis=1)

# 13. Element-wise operations
def element_wise_operations():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    addition = a + b
    multiplication = a * b
    
    return addition, multiplication

# 14. 3D array and sum along axes
def sum_3d_array():
    # Create a 3D array
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    # Compute sum along each axis
    sum_axis_0 = np.sum(arr, axis=0)  # Sum along first axis
    sum_axis_1 = np.sum(arr, axis=1)  # Sum along second axis
    sum_axis_2 = np.sum(arr, axis=2)  # Sum along third axis
    
    return sum_axis_0, sum_axis_1, sum_axis_2

# Section 5: Plotly

# 15. Interactive scatter plot
def interactive_scatter():
    x = np.random.rand(50)
    y = np.random.rand(50)
    sizes = np.random.randint(10, 50, 50)
    
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=sizes),
        text=[f'Point {i+1}<br>Size: {size}' for i, size in enumerate(sizes)],
        hoverinfo='text'
    ))
    
    fig.update_layout(title='Interactive Scatter Plot')
    fig.show()

# 16. 3D surface plot
def surface_plot():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
    fig.update_layout(title='3D Surface Plot: sin(sqrt(x^2 + y^2))')
    fig.show()

# Section 6: Advanced Python Functions, Loops, and Map

# 17. Filter list of dictionaries
def filter_dictionaries(list_of_dicts, key, threshold):
    return [d for d in list_of_dicts if d.get(key, 0) > threshold]

# 18. List comprehension with complex function
def complex_list_comprehension(numbers):
    return [np.sin(x)**2 + np.cos(x)**2 for x in numbers]

# 19. Recursive factorial
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

# 20. Filter even numbers
def filter_even_numbers(numbers):
    return list(filter(lambda x: x % 2 == 0, numbers))

# Main function to demonstrate usage
def main():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': ['X', 'Y', 'Z', 'X', np.nan]
    })
    
    # Demonstrate Pandas functions
    print("Pandas examples:")
    dropped, filled = handle_missing_data(df)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame with dropped rows:")
    print(dropped)
    print("\nDataFrame with filled values:")
    print(filled)
    print("\nTop 3 rows with highest values in column 'A':")
    print(top_3_rows(df, 'A'))
    print("\nGrouped sum of 'A' by 'C':")
    print(group_and_sum(df, ['C'], 'A'))
    print("\nNormalized column 'A':")
    print(normalize_column(df.copy(), 'A'))
    
    # Demonstrate Matplotlib functions
    custom_plot()
    subplot_example()
    secondary_y_axis()
    
    # Demonstrate Seaborn functions
    sns_df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    barplot_example(sns_df)
    countplot_example(sns_df)
    violin_plot(sns_df)
    improved_seaborn_plot(sns_df)
    
    # Demonstrate NumPy functions
    print("NumPy examples:")
    print(max_row_index(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    print(element_wise_operations())
    print(sum_3d_array())
    
    # Demonstrate Plotly functions
    interactive_scatter()
    surface_plot()
    
    # Demonstrate advanced Python functions
    list_of_dicts = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
    print("Advanced Python examples:")
    print(filter_dictionaries(list_of_dicts, 'a', 2))
    print(complex_list_comprehension([0, np.pi/4, np.pi/2]))
    print(factorial(5))
    print(filter_even_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

if __name__ == "__main__":
    main()