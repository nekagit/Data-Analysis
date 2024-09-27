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

# 1. Bar Plot: Average Salary by Department
plt.figure(figsize=(10,6))
sns.barplot(x='Department', y='Salary', data=df)
plt.xticks(rotation=45)
plt.title('Average Salary by Department')
plt.show()

# 2. Scatter Plot: Salary vs Performance Rating
plt.figure(figsize=(10,6))
sns.scatterplot(x='Salary', y='PerformanceRating', data=df, hue='Department')
plt.title('Salary vs Performance Rating')
plt.show()

# 3. Box Plot: Salary Distribution by Department
plt.figure(figsize=(10,6))
sns.boxplot(x='Department', y='Salary', data=df, palette='Set2')
plt.xticks(rotation=45)
plt.title('Salary Distribution by Department')
plt.show()

# 4. Count Plot: Number of Employees by Department
plt.figure(figsize=(10,6))
sns.countplot(x='Department', data=df, palette='Set1')
plt.xticks(rotation=45)
plt.title('Number of Employees by Department')
plt.show()

# 5. Line Plot: Salary Over Time (Start Dates)
plt.figure(figsize=(10,6))
sns.lineplot(x='StartDate', y='Salary', data=df, marker='o')
plt.title('Salary Over Time')
plt.show()

# 6. Heatmap: Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 7. Violin Plot: Salary Distribution by Department
plt.figure(figsize=(10,6))
sns.violinplot(x='Department', y='Salary', data=df)
plt.xticks(rotation=45)
plt.title('Violin Plot of Salary by Department')
plt.show()

# 8. Pair Plot: Relationships Between Numerical Features
sns.pairplot(df[['Salary', 'PerformanceRating']])
plt.title('Pair Plot of Numerical Features')
plt.show()

# 9. Histogram: Salary Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['Salary'], kde=True)
plt.title('Salary Distribution')
plt.show()

# 10. FacetGrid: Performance Rating by Project
g = sns.FacetGrid(df, col="Project", col_wrap=3)
g.map(sns.scatterplot, "Salary", "PerformanceRating")
plt.show()

# 11. Strip Plot: Performance Rating by Role
plt.figure(figsize=(10,6))
sns.stripplot(x='Role', y='PerformanceRating', data=df, jitter=True)
plt.xticks(rotation=45)
plt.title('Performance Rating by Role')
plt.show()
