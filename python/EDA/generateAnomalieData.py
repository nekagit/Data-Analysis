import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Setting random seed for reproducibility
np.random.seed(42)

# **Employee Data** with anomalies and duplicates
employee_data = pd.DataFrame({
    'EmployeeID': range(1, 101),
    'Name': [f'Employee{i}' for i in range(1, 101)],
    'Department': np.random.choice(['R&D', 'Engineering', 'Sales', 'HR', 'Finance'], 100),
    'Salary': np.random.randint(50000, 150000, 100),
    'Performance': np.random.uniform(2, 5, 100),
    'YearsOfExperience': np.random.randint(0, 20, 100),
    'SkillLevel': np.random.randint(1, 10, 100)
})

# Introduce anomalies
employee_data.loc[5, 'Salary'] = 9999999  # Extremely high salary
employee_data.loc[10, 'Performance'] = -5  # Negative performance value

# Add duplicates
employee_data = pd.concat([employee_data, employee_data.iloc[2:4]], ignore_index=True)

# Add missing values
employee_data.loc[8, 'Department'] = np.nan
employee_data.loc[20, 'SkillLevel'] = np.nan

# **Project Data** with anomalies and duplicates
project_data = pd.DataFrame({
    'ProjectID': range(1, 51),
    'ProjectName': [f'Project{i}' for i in range(1, 51)],
    'StartDate': pd.date_range(start='2018-01-01', periods=50),
    'Duration': np.random.randint(30, 365, 50),
    'Budget': np.random.randint(100000, 1000000, 50),
    'ActualCost': np.random.randint(80000, 1200000, 50),
    'Revenue': np.random.randint(150000, 2000000, 50),
    'ProjectManager': np.random.choice(employee_data['Name'], 50)
})

# Introduce anomalies
project_data.loc[3, 'Duration'] = -50  # Negative duration
project_data.loc[7, 'Budget'] = 99999999  # Unreasonably high budget

# Add duplicates
project_data = pd.concat([project_data, project_data.iloc[5:7]], ignore_index=True)

# Add missing values
project_data.loc[15, 'ProjectManager'] = np.nan

# **Financial Data** with anomalies and duplicates
start_date = datetime(2018, 1, 1)
dates = [start_date + timedelta(days=30 * i) for i in range(60)]
financial_data = pd.DataFrame({
    'Date': dates,
    'Revenue': np.random.randint(1000000, 5000000, 60),
    'Expenses': np.random.randint(800000, 4000000, 60),
    'Profit': np.random.randint(100000, 1000000, 60)
})

# Introduce anomalies
financial_data.loc[10, 'Revenue'] = -1000000  # Negative revenue
financial_data.loc[20, 'Expenses'] = 9999999  # Extremely high expense

# Add duplicates
financial_data = pd.concat([financial_data, financial_data.iloc[0:2]], ignore_index=True)

# Add missing values
financial_data.loc[5, 'Profit'] = np.nan

# **Customer Data** with anomalies and duplicates
customer_data = pd.DataFrame({
    'CustomerID': range(1, 201),
    'CustomerName': [f'Customer{i}' for i in range(1, 201)],
    'Satisfaction': np.random.uniform(2, 5, 200),
    'ContractValue': np.random.randint(10000, 1000000, 200),
    'ContractDuration': np.random.randint(6, 60, 200)
})

# Introduce anomalies
customer_data.loc[3, 'Satisfaction'] = 10  # Satisfaction beyond the valid range
customer_data.loc[50, 'ContractValue'] = -5000  # Negative contract value

# Add duplicates
customer_data = pd.concat([customer_data, customer_data.iloc[5:8]], ignore_index=True)

# Add missing values
customer_data.loc[12, 'ContractDuration'] = np.nan

# **Market Data** with anomalies and duplicates
market_data = pd.DataFrame({
    'Quarter': pd.date_range(start='2018-01-01', periods=20, freq='QE'),
    'MarketSize': np.random.randint(10000000, 50000000, 20),
    'MarketShare': np.random.uniform(0.05, 0.3, 20),
    'CompetitorShare': np.random.uniform(0.4, 0.7, 20)
})

# Introduce anomalies
market_data.loc[6, 'MarketShare'] = 1.5  # Market share exceeding 100%
market_data.loc[12, 'CompetitorShare'] = -0.2  # Negative competitor share

# Add duplicates
market_data = pd.concat([market_data, market_data.iloc[2:4]], ignore_index=True)

# Add missing values
market_data.loc[7, 'MarketSize'] = np.nan


# For this task, we'll need to create some dummy supply chain data
suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
delivery = pd.DataFrame({
'Supplier': suppliers,
'AverageDeliveryTime': np.random.uniform(1, 100, 5),
'DeliveryVariance': np.random.uniform(0, 2, 5)
})

# Ensure folder structure for each analysis task
def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

create_folder('data')

# Save to CSV
employee_data.to_csv('data/employee_data_with_anomalies.csv', index=False)
project_data.to_csv('data/project_data_with_anomalies.csv', index=False)
financial_data.to_csv('data/financial_data_with_anomalies.csv', index=False)
customer_data.to_csv('data/customer_data_with_anomalies.csv', index=False)
market_data.to_csv('data/market_data_with_anomalies.csv', index=False)
delivery.to_csv('data/delivery_data_with_anomalies.csv', index=False)

# Load the datasets
# employee_data = pd.read_csv('data/employee_data_with_anomalies.csv')
# project_data = pd.read_csv('data/project_data_with_anomalies.csv')
# financial_data = pd.read_csv('data/financial_data_with_anomalies.csv')
# customer_data = pd.read_csv('data/customer_data_with_anomalies.csv')
# market_data = pd.read_csv('data/market_data_with_anomalies.csv')
# delivery_data = pd.read_csv('data/delivery_data_with_anomalies.csv')


# Create Plot folder for images
# create_folder('plots')
# plt.savefig('plots/skill_gap_analysis.png')


