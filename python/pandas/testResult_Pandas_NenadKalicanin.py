import pandas as pd
import numpy as np

# Create the dataset
data = {
    'EmployeeID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'FirstName': ['John', 'Jane', 'Michael', 'Emily', 'David', 'Alice', 'Robert', 'Laura', 'James', 'Emma'],
    'LastName': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown', 'Davis', 'Wilson', 'Moore', 'Taylor', 'Anderson'],
    'Role': ['Software Engineer', 'Data Scientist', 'UX Designer', 'Product Manager', 'QA Engineer', 'DevOps Engineer', 'Backend Developer', 'Frontend Developer', 'HR Manager', 'Marketing Specialist'],
    'Department': ['Development', 'Data Science', 'Design', 'Management', 'Quality Assurance', 'Operations', 'Development', 'Development', 'Human Resources', 'Marketing'],
    'Salary': [70000, 75000, 68000, 80000, 65000, 72000, 71000, 69000, 73000, 67000],
    'StartDate': ['2022-01-15', '2021-06-01', '2023-03-10', '2020-11-01', '2022-07-15', '2021-09-20', '2023-02-05', '2022-10-25', '2020-12-01', '2021-08-15'],
    'Project': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 'Project Epsilon', 'Project Zeta', 'Project Alpha', 'Project Beta', 'NA', 'Project Gamma'],
    'PerformanceRating': [4.5, 4.7, 4.2, 4.6, 4.1, 4.3, 4.4, 4.0, 4.8, 4.2]
}

# Question 1: Primary data structure in Pandas for handling tabular data
df = pd.DataFrame(data)
print("Question 1: DataFrame example")
print(df.head())
print("\n" + "="*50 + "\n")

# Question 2: Reading a CSV file into a Pandas DataFrame
df.to_csv('employee_data.csv', index=False)
df_from_csv = pd.read_csv('employee_data.csv')
print("Question 2: Reading CSV example")
print(df_from_csv.head())
print("\n" + "="*50 + "\n")

# Question 3: Displaying the first 5 rows of a DataFrame
print("Question 3: First 5 rows of DataFrame")
print(df.head(5))
print("\n" + "="*50 + "\n")

# Question 4: Getting a summary of basic statistics for numeric columns
print("Question 4: Summary of basic statistics")
print(df.describe())
print("\n" + "="*50 + "\n")

# Question 5: Filtering DataFrame to show only rows where age > 30
df['Age'] = np.random.randint(25, 55, size=len(df))  # Adding an Age column for demonstration
print("Question 5: Filtering DataFrame")
print(df[df['Age'] > 30])
print("\n" + "="*50 + "\n")

# Question 6: Renaming columns of a DataFrame
new_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
df.columns = new_columns
print("Question 6: Renamed columns")
print(df.head())
print("\n" + "="*50 + "\n")

# Question 7: Removing rows with missing data
df.loc[0, 'A'] = np.nan  # Adding a NaN value for demonstration
print("Question 7: Removing rows with missing data")
print(df.dropna())
print("\n" + "="*50 + "\n")

# Question 8: Calculating the mean of values in a column
print("Question 8: Mean of 'F' column (Salary)")
print(df['F'].mean())
print("\n" + "="*50 + "\n")

# Question 9: Merging two DataFrames
df2 = pd.DataFrame({'A': [11, 12], 'K': ['New1', 'New2']})
merged_df = pd.merge(df, df2, on='A', how='outer')
print("Question 9: Merged DataFrame")
print(merged_df)
print("\n" + "="*50 + "\n")

# Question 10: Grouping DataFrame and computing sum
print("Question 10: Grouping by 'E' (Department) and summing 'F' (Salary)")
print(df.groupby('E')['F'].sum())