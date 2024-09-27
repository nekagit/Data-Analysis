import pandas as pd


employee_data = pd.read_csv("startup_data.csv")

employee_data.columns = employee_data.columns.str.strip()

print(employee_data.head())