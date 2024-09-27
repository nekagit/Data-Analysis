
# basics.py
import pandas as pd

def run_basics_tutorial(df):
    print("\n=== Pandas Basics ===")
    
    # Creating a DataFrame
    print("1. Creating a DataFrame:")
    print(df.head(2))
    
    # Accessing columns
    print("\n2. Accessing columns:")
    print(df['Salary'])
    
    # Accessing rows
    print("\n3. Accessing rows:")
    print(df.loc[2])
    
    # DataFrame info and description
    print("\n4. DataFrame info:")
    df.info()
    print("\n5. DataFrame description:")
    print(df.describe())