import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Generate sample data
n_samples = len(date_range)
data = {
    'Date': date_range,
    'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C'], n_samples),
    'Sales': np.random.randint(0, 100, n_samples),
    'Revenue': np.random.uniform(100, 1000, n_samples),
    'Customer_Age': np.random.randint(18, 65, n_samples),
    'Customer_Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'Marketing_Spend': np.random.uniform(50, 500, n_samples),
    'Customer_Satisfaction': np.random.randint(1, 6, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Introduce some missing values
df.loc[np.random.choice(df.index, 50), 'Sales'] = np.nan
df.loc[np.random.choice(df.index, 50), 'Customer_Age'] = np.nan

# Add some outliers
df.loc[np.random.choice(df.index, 10), 'Revenue'] = df['Revenue'].max() * 5

# Save to CSV
df.to_csv('startup_sales_data.csv', index=False)

print("Data generated and saved to 'startup_sales_data.csv'")