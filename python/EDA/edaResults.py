import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# pip install pandas numpy plotly kaleido
# Task 1: Data Cleaning and Preprocessing

employee_df = pd.read_csv('employee_data.csv')
project_df = pd.read_csv('project_data.csv')

employee_df.columns = employee_df.columns.str.strip()
project_df.columns = project_df.columns.str.strip()

# Convert date columns
employee_df['HiringDate'] = pd.to_datetime(employee_df['HiringDate'])
project_df['StartDate'] = pd.to_datetime(project_df['StartDate'])
project_df['EndDate'] = pd.to_datetime(project_df['EndDate'])

# Create derived columns
employee_df['EmployeeCost'] = employee_df['Salary'] + employee_df['Benefits']
project_df['ProjectROI'] = (project_df['ProjectRevenue'] - project_df['ProjectCost']) / project_df['ProjectCost']

# Task 2: Calculate Derived Metrics
employee_df['AnnualBudgetSurplus'] = (employee_df['MonthlyBudget'] - employee_df['MonthlyExpenses']) * 12
employee_df['AnnualProfitMargin'] = (employee_df['MonthlyRevenue'] - employee_df['MonthlyExpenses']) * 12
project_df['ProjectDuration'] = (project_df['EndDate'] - project_df['StartDate']).dt.days
project_df['DailyRevenue'] = project_df['ProjectRevenue'] / project_df['ProjectDuration']

# Print important data
print("Employee Data Summary:")
print(employee_df.describe())
print("\nProject Data Summary:")
print(project_df.describe())

# Task 3: Department Overview Dashboard
def create_department_overview():
    dept_counts = employee_df['Department'].value_counts()
    avg_salaries = employee_df.groupby('Department')['Salary'].mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=dept_counts.index, y=dept_counts.values, name="Employee Count"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=avg_salaries.index, y=avg_salaries.values, name="Average Salary", mode="lines+markers"),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Department Overview",
        xaxis_title="Department",
    )

    fig.update_yaxes(title_text="Employee Count", secondary_y=False)
    fig.update_yaxes(title_text="Average Salary", secondary_y=True)

    fig.write_image("department_overview.png")
    print("\nDepartment Overview:")
    print(dept_counts)
    print("\nAverage Salaries by Department:")
    print(avg_salaries)

# Task 4: Salary Analysis
def create_salary_analysis():
    fig = go.Figure()

    for dept in employee_df['Department'].unique():
        dept_data = employee_df[employee_df['Department'] == dept]
        fig.add_trace(go.Box(y=dept_data['Salary'], name=dept, boxpoints='all', jitter=0.3, pointpos=-1.8))

    fig.update_layout(title_text="Salary Distribution by Department",
                      xaxis_title="Department",
                      yaxis_title="Salary")

    fig.write_image("salary_analysis.png")
    print("\nSalary Statistics by Department:")
    print(employee_df.groupby('Department')['Salary'].describe())

# Task 5: Interactive Employee Performance Dashboard
def create_performance_dashboard():
    fig = px.scatter(employee_df, x="PerformanceRating", y="Salary", color="YearsOfExperience",
                     hover_data=["FirstName", "LastName", "Department"],
                     title="Employee Performance vs Salary")
    
    fig.update_layout(xaxis_title="Performance Rating",
                      yaxis_title="Salary",
                      coloraxis_colorbar_title="Years of Experience")

    fig.write_image("performance_dashboard.png")
    print("\nPerformance Rating vs Salary Correlation:")
    print(employee_df['PerformanceRating'].corr(employee_df['Salary']))

# Task 6: Project Profitability Analysis
def create_project_profitability():
    fig = px.scatter(project_df, x="ProjectDuration", y="ProjectROI", color="ProjectStatus",
                     hover_data=["ProjectName", "AssignedEmployeeID", "ProjectRevenue"],
                     title="Project Profitability Analysis")
    
    fig.update_layout(xaxis_title="Project Duration (days)",
                      yaxis_title="Project ROI")

    fig.write_image("project_profitability.png")
    print("\nProject Profitability Summary:")
    print(project_df[['ProjectDuration', 'ProjectROI', 'ProjectStatus']].describe())

# Task 7: Budget Optimization Analysis
def create_budget_analysis():
    dept_finance = employee_df.groupby('Department').agg({
        'MonthlyBudget': 'sum',
        'MonthlyRevenue': 'sum',
        'MonthlyExpenses': 'sum'
    }).reset_index()

    dept_finance['ProfitMargin'] = (dept_finance['MonthlyRevenue'] - dept_finance['MonthlyExpenses']) / dept_finance['MonthlyRevenue']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=dept_finance['Department'], y=dept_finance['MonthlyBudget'], name="Budget", offsetgroup=0),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=dept_finance['Department'], y=dept_finance['MonthlyRevenue'], name="Revenue", offsetgroup=0),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=dept_finance['Department'], y=dept_finance['MonthlyExpenses'], name="Expenses", offsetgroup=0),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=dept_finance['Department'], y=dept_finance['ProfitMargin'], name="Profit Margin", mode="lines+markers"),
        secondary_y=True,
    )

    fig.update_layout(title_text="Budget, Revenue, and Expenses by Department",
                      barmode='group',
                      xaxis_title="Department")

    fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit Margin", secondary_y=True)

    fig.write_image("budget_analysis.png")
    print("\nDepartment Finance Summary:")
    print(dept_finance)

# Task 8: Employee Retention Risk Assessment
def create_retention_risk_assessment():
    fig = go.Figure(data=go.Scatter3d(
        x=employee_df['YearsOfExperience'],
        y=employee_df['PerformanceRating'],
        z=employee_df['Salary'],
        mode='markers',
        marker=dict(
            size=5,
            color=employee_df['RetentionRisk'],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=employee_df['FirstName'] + ' ' + employee_df['LastName'],
        hoverinfo='text'
    ))

    fig.update_layout(title="Employee Retention Risk Assessment",
                      scene=dict(
                          xaxis_title="Years of Experience",
                          yaxis_title="Performance Rating",
                          zaxis_title="Salary"
                      ))

    fig.write_image("retention_risk_assessment.png")
    print("\nRetention Risk Summary:")
    print(employee_df.groupby('RetentionRisk').agg({
        'YearsOfExperience': 'mean',
        'PerformanceRating': 'mean',
        'Salary': 'mean'
    }))

# Task 9: Time Series Analysis of Financial Metrics
def create_financial_trends():
    monthly_data = employee_df.groupby('Department').agg({
        'MonthlyRevenue': 'sum',
        'MonthlyExpenses': 'sum'
    }).reset_index()
    monthly_data['MonthlyProfit'] = monthly_data['MonthlyRevenue'] - monthly_data['MonthlyExpenses']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_data['Department'], y=monthly_data['MonthlyRevenue'], mode='lines+markers', name='Revenue'))
    fig.add_trace(go.Scatter(x=monthly_data['Department'], y=monthly_data['MonthlyExpenses'], mode='lines+markers', name='Expenses'))
    fig.add_trace(go.Scatter(x=monthly_data['Department'], y=monthly_data['MonthlyProfit'], mode='lines+markers', name='Profit'))

    fig.update_layout(title="Financial Trends by Department",
                      xaxis_title="Department",
                      yaxis_title="Amount ($)")

    fig.write_image("financial_trends.png")
    print("\nFinancial Trends Summary:")
    print(monthly_data)

# Task 10: Comprehensive Business Health Dashboard
def create_comprehensive_dashboard():
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Department Overview", "Salary Analysis", "Employee Performance",
                        "Project Profitability", "Budget Analysis", "Financial Trends"),
        specs=[[{"type": "xy", "secondary_y": True}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "secondary_y": True}, {"type": "xy"}]]
    )

    # Department Overview
    dept_counts = employee_df['Department'].value_counts()
    avg_salaries = employee_df.groupby('Department')['Salary'].mean()
    fig.add_trace(go.Bar(x=dept_counts.index, y=dept_counts.values, name="Employee Count"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=avg_salaries.index, y=avg_salaries.values, name="Average Salary", mode="lines+markers"), row=1, col=1, secondary_y=True)

    # Salary Analysis
    for dept in employee_df['Department'].unique():
        dept_data = employee_df[employee_df['Department'] == dept]
        fig.add_trace(go.Box(y=dept_data['Salary'], name=dept, boxpoints='all', jitter=0.3, pointpos=-1.8), row=1, col=2)

    # Employee Performance
    fig.add_trace(go.Scatter(x=employee_df["PerformanceRating"], y=employee_df["Salary"], mode='markers',
                             marker=dict(color=employee_df["YearsOfExperience"], colorscale="Viridis", showscale=True),
                             text=employee_df['FirstName'] + ' ' + employee_df['LastName'],
                             hoverinfo='text'), row=2, col=1)

    # Project Profitability
    fig.add_trace(go.Scatter(x=project_df["ProjectDuration"], y=project_df["ProjectROI"], mode='markers',
                             marker=dict(color=project_df["ProjectStatus"].astype('category').cat.codes, showscale=True),
                             text=project_df['ProjectName'],
                             hoverinfo='text'), row=2, col=2)

    # Budget Analysis
    dept_finance = employee_df.groupby('Department').agg({
        'MonthlyBudget': 'sum',
        'MonthlyRevenue': 'sum',
        'MonthlyExpenses': 'sum'
    }).reset_index()
    dept_finance['ProfitMargin'] = (dept_finance['MonthlyRevenue'] - dept_finance['MonthlyExpenses']) / dept_finance['MonthlyRevenue']
    
    fig.add_trace(go.Bar(x=dept_finance['Department'], y=dept_finance['MonthlyBudget'], name="Budget"), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=dept_finance['Department'], y=dept_finance['MonthlyRevenue'], name="Revenue"), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=dept_finance['Department'], y=dept_finance['MonthlyExpenses'], name="Expenses"), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=dept_finance['Department'], y=dept_finance['ProfitMargin'], name="Profit Margin", mode="lines+markers"), row=3, col=1, secondary_y=True)

    # Financial Trends
    monthly_data = employee_df.groupby('Department').agg({
        'MonthlyRevenue': 'sum',
        'MonthlyExpenses': 'sum'
    }).reset_index()
    monthly_data['MonthlyProfit'] = monthly_data['MonthlyRevenue'] - monthly_data['MonthlyExpenses']

    fig.add_trace(go.Scatter(x=monthly_data['Department'], y=monthly_data['MonthlyRevenue'], mode='lines+markers', name='Revenue'), row=3, col=2)
    fig.add_trace(go.Scatter(x=monthly_data['Department'], y=monthly_data['MonthlyExpenses'], mode='lines+markers', name='Expenses'), row=3, col=2)
    fig.add_trace(go.Scatter(x=monthly_data['Department'], y=monthly_data['MonthlyProfit'], mode='lines+markers', name='Profit'), row=3, col=2)

    fig.update_layout(height=1800, width=1200, title_text="Comprehensive Business Health Dashboard")
    fig.write_image("comprehensive_dashboard.png")
    print("\nComprehensive Business Health Summary:")
    print(employee_df.describe())
    print("\n")
    print(project_df.describe())

# Execute all functions
create_department_overview()
create_salary_analysis()
create_performance_dashboard()
create_project_profitability()
create_budget_analysis()
create_retention_risk_assessment()
create_financial_trends()
create_comprehensive_dashboard()

print("\nAll visualizations have been saved as PNG files in the current directory.")