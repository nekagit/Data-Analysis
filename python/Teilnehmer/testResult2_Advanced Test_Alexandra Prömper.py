### Section 1: Pandas

#1. Explain how to handle missing data in a Pandas DataFrame. Write a code snippet demonstrating two different methods to deal with missing values.

#2. Given a DataFrame, how would you find and return the top 3 rows with the highest values in a specific column while ignoring NaN values? Provide a code example.

#3. Describe the purpose of the `groupby()` function in Pandas. Write a code snippet that demonstrates how to group data by two columns and calculate the sum of another column.

#4. Write a function that takes a DataFrame and a column name as inputs, normalizes the values in that column, and returns the modified DataFrame.

#5. Explain how to merge two DataFrames in Pandas. Write a code snippet to demonstrate merging on a common key with an inner join.



#1)
    # Missing Data in Python can be extracted (droped) or renamed

import pandas as pd
import numpy as np

# Create a DataFrame with missing values
data = {'name': ['Alex', 'Diana', 'Daniela'],
        'age': [35, None, 35],
        'weight': [69.0, 67.0, None]}

df = pd.DataFrame(data)
#print(df)

# 1. drop the NaN

#df_ohne_NaN = df.dropna(axis=1)
##print(df_ohne_NaN)

# 2. rename with a constant (0)

df_filled = df.fillna(0)
#print(df_filled)


#2) 

# Create a DataFrame with missing values
data1 = {'name': ['Alex', 'Diana', 'Daniela', 'Michael', 'Carola'],
        'age': [35, None, 36, 61, 60],
        'weight': [69.0, 67.0, None, 93, 74]}

df1 = pd.DataFrame(data1)
#print(data1)

# top 3 Alter
top_3 = df1.nlargest(3, 'age') #nlargest: return top 3 with values of Age ignoring NaN
#print(top_3)


#3) groupby()` function in Pandas: Mit der Funktion groupby() können bestimmte Werte nach bestimmten festgelegten Kriterien gruppiert werden

data2 = {'name': ['Alex', 'Diana', 'Daniela', 'Michael', 'Carola'],
        'age': [35, None, 36, 61, 60],
        'weight': [69.0, 67.0, None, 93, 74]}

df2 = pd.DataFrame(data2)

# mit group by() nach Name gruppieren

gruppiert = df2.groupby('name')
#print(gruppiert)


#4) 

data3 = {'name': ['Alex', 'Diana', 'Daniela', 'Michael', 'Carola'],
        'age': [35, None, 36, 61, 60],
        'weight': [69.0, 67.0, None, 93, 74]}

df3 = pd.DataFrame(data3)

def normalize_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    # Drop NaN values for calculation, but keep them in the original column
    column_no_nan = df[column_name].dropna()
    
    if len(column_no_nan) == 0:
        return df  # Return original dataframe if all values are NaN
    
    min_val = column_no_nan.min()
    max_val = column_no_nan.max()
    
    # Avoid division by zero
    if min_val == max_val:
        df[f'{column_name}_normalized'] = 1  # or 0, depending on your preference
    else:
        df[f'{column_name}_normalized'] = (df[column_name] - min_val) / (max_val - min_val)
    
    return df

# Apply the function
normalize_df3 = normalize_column(df3, 'age')
print(normalize_df3)

# Diese Normalisierung ermoeglicht eine einsicht was der kleinste wert und was der groesste wert ist die durch 0 und 1 repraesentiert werden und dann die realtionen der werte dazwischen
# Diana: NaN (remains NaN after normalization) 
# Daniela (36): (36 - 35) / (61 - 35) = 1 / 26 ≈ 0.038462
# Michael (61): (61 - 35) / (61 - 35) = 26 / 26 = 1
# Carola (60): (60 - 35) / (61 - 35) = 25 / 26 ≈ 0.961538

#5)

#Mit merge können zwei 2 Panda Dataframes kombineirt werden. Mit Inner Join Merge werden die Zeilen auf Grundlage einer angegebenen Spalte abgeglichen und ein neuer DataFrame mit gemeinsamen Werten ausgegeben.

#Use df2 and create a second dataframe (df4) for merging

data4 = {'name': ['Alex', 'Diana', 'Daniela', 'Michael', 'Carola'],
        'marks':[80, 90, 75, 88, 59]} # erreichete Punktzahl im Test von 100 möglichen Punkten

df4 = pd.DataFrame(data4)

#print(df2, df4)

#merge the dataframes
df_merged = df2.merge(df4[['name', 'marks']])
#print(df_merged)


### Section 2: Matplotlib

#6. How can you customize the aesthetics of a Matplotlib plot? Write a code example that includes at least three different customizations.

#7. Write a Matplotlib code snippet to create a subplot with three different types of plots (line, bar, and scatter) in a single figure, each with its own title.

#8. Explain how to add a secondary y-axis to a plot. Write a code snippet that demonstrates this with a sample dataset.


#6) Customization with e.g. Title, xLabel und yLabel and customization: marker, grid and axis labels

#import necessary Bib.
import matplotlib.pyplot as plt

#use data from dataframe df4
data4 = {'name': ['Alex', 'Diana', 'Daniela', 'Michael', 'Carola'],
        'marks':[80, 90, 75, 88, 59]} 

# Zeit ist um








    
    

