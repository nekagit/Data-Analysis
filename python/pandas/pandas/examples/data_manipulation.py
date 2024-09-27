
# data_manipulation.py
import pandas as pd

def run_data_manipulation_tutorial(df):
    print("\n=== Data Manipulation ===")
    
    # Filtering data
    print("1. Filtering data (Salary > 40000):")
    high_salary = df[df['Salary'] > 40000]
    print(high_salary)
    
    # Sorting data
    print("\n2. Sorting data by Last Name:")
    sorted_df = df.sort_values(by='Last Name')
    print(sorted_df)
    
    # Adding a new column
    print("\n3. Adding a new column (Salary in EUR):")
    df['Salary_EUR'] = df['Salary'] * 0.85
    print(df[['Salary', 'Salary_EUR']])
    
    # Renaming columns
    print("\n4. Renaming columns:")
    df_renamed = df.rename(columns={'First Name': 'FirstName', 'Last Name': 'LastName'})
    print(df_renamed.columns)
    
    # Handling missing data
    print("\n5. Handling missing data:")
    df.loc[2, 'Salary'] = None
    print("DataFrame with missing value:\n", df)
    print("\nDataFrame after filling missing values:")
    print(df.fillna(0))
