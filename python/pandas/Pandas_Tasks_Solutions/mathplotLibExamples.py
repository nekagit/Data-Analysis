import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the dataset
data = {
    'EmployeeID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'FirstName': ['John', 'Jane', 'Michael', 'Emily', 'David', 'Alice', 'Robert', 'Laura', 'James', 'Emma'],
    'LastName': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown', 'Davis', 'Wilson', 'Moore', 'Taylor', 'Anderson'],
    'Role': ['Software Engineer', 'Data Scientist', 'UX Designer', 'Product Manager', 'QA Engineer', 'DevOps Engineer', 'Backend Developer', 'Frontend Developer', 'HR Manager', 'Marketing Specialist'],
    'Department': ['Development', 'Data Science', 'Design', 'Management', 'Quality Assurance', 'Operations', 'Development', 'Development', 'Human Resources', 'Marketing'],
    'Salary': [70000, 75000, 68000, 80000, 65000, 72000, 71000, 69000, 73000, 67000],
    'StartDate': ['2022-01-15', '2021-06-01', '2023-03-10', '2020-11-01', '2022-07-15', '2021-09-20', '2023-02-05', '2022-10-25', '2020-12-01', '2021-08-15'],
    'Project': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 'Project Epsilon', 'Project Zeta', 'Project Alpha', 'Project Beta', 'NA', 'Project Gamma'],
    'PerformanceRating': [4.5, 4.7, 4.2, 4.6, 4.1, 4.3, 4.4, 4.0, 4.8, 4.2]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert StartDate to datetime
df['StartDate'] = pd.to_datetime(df['StartDate'])

# Matplotlib Visualizations

# 1. Bar Plot: Average Salary by Department
plt.figure(figsize=(10, 6))
avg_salary = df.groupby('Department')['Salary'].mean()
avg_salary.plot(kind='bar', color='skyblue')
plt.xticks(rotation=45)
plt.title('Average Salary by Department')
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.tight_layout()
plt.show()

# 2. Scatter Plot: Salary vs Performance Rating
plt.figure(figsize=(10, 6))
for dept in df['Department'].unique():
    subset = df[df['Department'] == dept]
    plt.scatter(subset['Salary'], subset['PerformanceRating'], label=dept)
plt.title('Salary vs Performance Rating')
plt.xlabel('Salary')
plt.ylabel('Performance Rating')
plt.legend(title='Department')
plt.show()

# 3. Box Plot: Salary Distribution by Department
plt.figure(figsize=(10, 6))
departments = df['Department'].unique()
data = [df[df['Department'] == dept]['Salary'] for dept in departments]
plt.boxplot(data, labels=departments, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Salary Distribution by Department')
plt.xticks(rotation=45)
plt.xlabel('Department')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

# 4. Count Plot: Number of Employees by Department
plt.figure(figsize=(10, 6))
df['Department'].value_counts().plot(kind='bar', color='lightgreen')
plt.xticks(rotation=45)
plt.title('Number of Employees by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.show()

# 5. Line Plot: Salary Over Time (Start Dates)
plt.figure(figsize=(10, 6))
plt.plot(df['StartDate'], df['Salary'], marker='o', linestyle='-')
plt.title('Salary Over Time')
plt.xlabel('Start Date')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()
plt.show()

# Seaborn Visualizations

# 1. Bar Plot: Average Salary by Department
plt.figure(figsize=(10, 6))
sns.barplot(x='Department', y='Salary', data=df)
plt.xticks(rotation=45)
plt.title('Average Salary by Department')
plt.tight_layout()
plt.show()

# 2. Scatter Plot: Salary vs Performance Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary', y='PerformanceRating', data=df, hue='Department')
plt.title('Salary vs Performance Rating')
plt.show()

# 3. Box Plot: Salary Distribution by Department
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Salary', data=df, palette='Set2')
plt.xticks(rotation=45)
plt.title('Salary Distribution by Department')
plt.show()

# 4. Count Plot: Number of Employees by Department
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', data=df, palette='Set1')
plt.xticks(rotation=45)
plt.title('Number of Employees by Department')
plt.show()

# 5. Line Plot: Salary Over Time (Start Dates)
plt.figure(figsize=(10, 6))
sns.lineplot(x='StartDate', y='Salary', data=df, marker='o')
plt.title('Salary Over Time')
plt.show()
