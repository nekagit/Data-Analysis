
# advanced.py
import pandas as pd
import numpy as np

def run_advanced_tutorial(df):
    print("\n=== Advanced Pandas Techniques ===")
    
    # Merging DataFrames
    print("1. Merging DataFrames:")
    bonus_data = pd.DataFrame({
        'Employee ID': [101, 102, 103, 104, 105],
        'Bonus': [5000, 3000, 4000, 3500, 6000]
    })
    merged_df = pd.merge(df, bonus_data, on='Employee ID')
    print(merged_df[['Employee ID', 'First Name', 'Salary', 'Bonus']])
    
    # Reshaping data with melt
    print("\n2. Reshaping data with melt:")
    melted_df = pd.melt(df, id_vars=['Employee ID', 'First Name', 'Last Name'], 
                        value_vars=['Salary', 'Salary_EUR'], 
                        var_name='Salary Type', value_name='Amount')
    print(melted_df.head(10))
    
    # TODO
    # Time series analysis
    print("\n3. Time series analysis:")
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df.set_index('Start Date', inplace=True)
    monthly_hires = df.resample('ME').size()
    print("Monthly hires:")
    print(monthly_hires)
    
    # Window functions
    print("\n4. Window functions (Rolling average of Salary):")
    df.sort_index(inplace=True)
    df['Rolling Avg Salary'] = df['Salary'].rolling(window=3, min_periods=1).mean()
    print(df[['Salary', 'Rolling Avg Salary']])
    
    # Categorical data
    print("\n5. Working with categorical data:")
    df['Department'] = pd.Categorical(df['Department'])
    print("Department categories:")
    print(df['Department'].cat.categories)
    
    # Applying custom functions
    print("\n6. Applying custom functions:")
    def salary_grade(salary):
        if salary < 40000:
            return 'Low'
        elif salary < 50000:
            return 'Medium'
        else:
            return 'High'
    
    df['Salary Grade'] = df['Salary'].apply(salary_grade)
    print(df[['First Name', 'Salary', 'Salary Grade']])