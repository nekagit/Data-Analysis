
# visualization.py
import pandas as pd
import matplotlib.pyplot as plt

def run_visualization_tutorial(df):
    print("\n=== Data Visualization ===")
    
    # Bar plot
    print("1. Bar plot of Salaries:")
    plt.figure(figsize=(10, 5))
    plt.bar(df['First Name'], df['Salary'])
    plt.title('Employee Salaries')
    plt.xlabel('Employee')
    plt.ylabel('Salary')
    plt.savefig('salary_bar_plot.png')
    plt.close()
    print("Bar plot saved as 'salary_bar_plot.png'")
    
    # Scatter plot
    print("\n2. Scatter plot of Salary vs Years of Service:")
    df['Years of Service'] = (pd.to_datetime('2023-01-01') - pd.to_datetime(df['Start Date'])).dt.days / 365
    plt.figure(figsize=(10, 5))
    plt.scatter(df['Years of Service'], df['Salary'])
    plt.title('Salary vs Years of Service')
    plt.xlabel('Years of Service')
    plt.ylabel('Salary')
    plt.savefig('salary_scatter_plot.png')
    plt.close()
    print("Scatter plot saved as 'salary_scatter_plot.png'")
    
    # Box plot
    print("\n3. Box plot of Salaries by Department:")
    plt.figure(figsize=(10, 5))
    df.boxplot(column='Salary', by='Department')
    plt.title('Salary Distribution by Department')
    plt.suptitle('')
    plt.savefig('salary_box_plot.png')
    plt.show()

    plt.close()
    print("Box plot saved as 'salary_box_plot.png'")
