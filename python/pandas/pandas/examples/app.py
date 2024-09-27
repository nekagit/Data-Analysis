# main.py
import pandas as pd
from basics import run_basics_tutorial
from data_manipulation import run_data_manipulation_tutorial
from analysis import run_analysis_tutorial
from visualization import run_visualization_tutorial
from advanced import run_advanced_tutorial

def main():
    # Create a sample DataFrame to be used across all tutorials
    data = {
        'Employee ID': [101, 102, 103, 104, 105],
        'First Name': ['John', 'Jane', 'Michael', 'Emily', 'David'],
        'Last Name': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown'],
        'Department': ['Marketing', 'Sales', 'Finance', 'HR', 'IT'],
        'Position': ['Manager', 'Associate', 'Analyst', 'Coordinator', 'Developer'],
        'Salary': [50000, 35000, 45000, 40000, 55000],
        'Start Date': ['2020-01-15', '2019-05-01', '2021-03-10', '2018-11-01', '2022-02-15']
    }
    # data = pd.read_csv('employee_data.csv')
    df = pd.DataFrame(data)
    
    print("Welcome to the Pandas Tutorial!")
    
    run_basics_tutorial(df)
    run_data_manipulation_tutorial(df)
    run_analysis_tutorial(df)
    run_visualization_tutorial(df)
    run_advanced_tutorial(df)
    
    print("\nTutorial completed. Thank you for learning Pandas!")

if __name__ == "__main__":
    main()

