import pandas as pd

# 1. DataFrame Creation
# Create DataFrames from dictionaries, lists, or NumPy arrays:
data = {
    'Employee ID': [101, 102, 103, 104, 105],
    'First Name': ['John', 'Jane', 'Michael', 'Emily', 'David'],
    'Last Name': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown'],
    'Department': ['Marketing', 'Sales', 'Finance', 'HR', 'IT'],
    'Position': ['Manager', 'Associate', 'Analyst', 'Coordinator', 'Developer'],
    'Salary': [50000, 35000, 45000, 40000, 55000],
    'Start Date': ['2020-01-15', '2019-05-01', '2021-03-10', '2018-11-01', '2022-02-15']
}
df = pd.DataFrame(data)

# 2. Data Inspection
# View the top/bottom rows of a DataFrame:
print(df.head())
print(df.tail())

# Get DataFrame info (data types, missing values):
df.info()
df.describe()

# Check the shape (rows, columns):
print(df.shape)

# Get unique values for a column:
print(df['Department'].unique())

# 3. Selection and Filtering
# Select columns:
print(df['First Name'])
print(df[['First Name', 'Department']])

# Filter rows based on conditions:
print(df[df['Salary'] > 40000])

# Select specific rows/columns by index:
print(df.iloc[0])
print(df.loc[df['Department'] == 'Marketing'])

# 4. Sorting
# Sort values by column(s):
df_sorted = df.sort_values(by='Salary', ascending=False)
print(df_sorted)

# 5. Aggregation and Grouping
# Group by 'Department' and calculate the mean only for numeric columns
df_numeric = df.select_dtypes(include=['number'])
grouped_mean = df_numeric.groupby(df['Department']).mean()

print(grouped_mean)

# 6. Merging and Joining
# Merge two DataFrames on a key:
bonus_data = {
    'Employee ID': [101, 102, 103, 104],
    'Bonus': [5000, 3000, 4000, 2500]
}
bonus_df = pd.DataFrame(bonus_data)
merged_df = pd.merge(df, bonus_df, on='Employee ID')
print(merged_df)

# Concatenate DataFrames (rows/columns):
df_concatenated = pd.concat([df, bonus_df], axis=1)
print(df_concatenated)

# 7. Missing Data Handling
# Check for missing values:
print(df.isnull().sum())

# Fill missing values:
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Drop missing values:
df.dropna()

# 8. Data Transformation
# Apply functions to DataFrame columns:
df['Adjusted Salary'] = df['Salary'].apply(lambda x: x + 1000)
print(df)

# Apply a function to the entire DataFrame or group of rows/columns:
df_transformed = df.applymap(lambda x: x if isinstance(x, (int, float)) else str(x).upper())
print(df_transformed)

# 9. String Operations
# String manipulations (e.g., split, lower, contains):
df['First Name'] = df['First Name'].str.lower()
print(df['First Name'])

# 10. Pivot Tables
# Create pivot tables:
pivot = df.pivot_table(values='Salary', index='Department', columns='Position', aggfunc='mean')
print(pivot)

# 11. Reshaping Data
# Stack/unstack a DataFrame:
stacked_df = df.stack()
print(stacked_df)
unstacked_df = stacked_df.unstack()
print(unstacked_df)

# Melt DataFrame (long format):
melted_df = pd.melt(df, id_vars=['First Name'], value_vars=['Salary'])
print(melted_df)

# 12. Time Series Analysis
# Convert strings to datetime:
df['Start Date'] = pd.to_datetime(df['Start Date'])
print(df['Start Date'])

# Set a column as the index (useful for time series):
df.set_index('Start Date', inplace=True)
print(df)

# 13. Exporting and Importing Data
# Read from/write to CSV, Excel, etc.:
df.to_csv('employees.csv')
df_from_csv = pd.read_csv('employees.csv')
print(df_from_csv)

# 14. Visualization
# Plot basic visualizations (line, bar, scatter, etc.):
df.plot(kind='bar', y='Salary', x='First Name', title='Salaries of Employees')

# 15. Advanced Filtering
# Filter rows based on multiple conditions:
filtered_df = df[(df['Salary'] > 45000) & (df['Department'] == 'Marketing')]
print(filtered_df)

# 16. Removing Duplicates
# Remove duplicates from a DataFrame:
df_unique = df.drop_duplicates()
print(df_unique)

# 17. Handling Categorical Data
# Convert a column to categorical data type:
df['Department'] = df['Department'].astype('category')
print(df['Department'].dtypes)

# 18. Mathematical Operations
# Perform element-wise operations:
df['Salary + Bonus'] = df['Salary'] + 1000
print(df['Salary + Bonus'])

# 19. Indexing and Slicing
# Set/reset the index of a DataFrame:
df_reset = df.reset_index()
print(df_reset)

# 20. Rolling Window Calculations
# Apply a rolling window operation (e.g., moving average):
df['Rolling Mean'] = df['Salary'].rolling(window=3).mean()
print(df['Rolling Mean'])
