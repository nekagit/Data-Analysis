

# analysis.py
import pandas as pd

def run_analysis_tutorial(df):
    print("\n=== Data Analysis ===")
    
    # Grouping and aggregation
    print("1. Grouping by Department and calculating mean Salary:")
    dept_salary = df.groupby('Department')['Salary'].mean()
    print(dept_salary)
    
    # Multiple aggregations
    print("\n2. Multiple aggregations:")
    agg_data = df.groupby('Department').agg({
        'Salary': ['mean', 'min', 'max'],
        'Employee ID': 'count'
    }).reset_index()
    print(agg_data)
    
    # Pivot tables
    print("\n3. Pivot table (Average Salary by Department and Position):")
    pivot = pd.pivot_table(df, values='Salary', index='Department', columns='Position', aggfunc='mean')
    print(pivot)
    
    # Correlation analysis
    print("\n4. Correlation analysis:")
    df['Years of Service'] = (pd.to_datetime('2023-01-01') - pd.to_datetime(df['Start Date'])).dt.days / 365
    correlation = df[['Salary', 'Years of Service']].corr()
    print(correlation)