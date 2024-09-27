import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Read the CSV file
df = pd.read_csv('startup_data.csv')
df.columns = df.columns.str.strip()

# Convert StartDate to datetime

df['StartDate'] = pd.to_datetime(df['StartDate'])

# Sort the dataframe by StartDate
df = df.sort_values('StartDate')

# Ensure numeric columns are properly typed
numeric_columns = ['Salary', 'PerformanceRating', 'Age', 'YearsOfExperience', 'Bonuses', 'WorkHoursPerWeek', 'VacationDaysTaken', 'TrainingHours', 'TeamSize', 'ClientSatisfactionScore']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Task 1: Basic Scatter Plot
scatter_fig = px.scatter(df, x="Salary", y="PerformanceRating", color="Department",
                        title="Scatter Plot of Salary vs Performance Rating by Department")
scatter_fig.show()

# Task 2: Line Chart for Time-Series Data
df['StartDate'] = pd.to_datetime(df['StartDate'])
line_fig = px.line(df, x='StartDate', y='Salary', title="Line Chart of Salary Growth Over Time")
line_fig.show()

# Task 3: Bar Chart Comparison
bar_fig = px.bar(df, x='Department', y='PerformanceRating', title="Bar Chart: Average Performance by Department")
bar_fig.show()

# Task 4: Pie Chart for Proportions
pie_fig = px.pie(df, names='Department', values='Salary', title="Pie Chart: Salary Distribution by Department")
pie_fig.show()

# Task 5: 3D Scatter Plot
scatter_3d_fig = px.scatter_3d(df, x='Salary', y='PerformanceRating', z='WorkHoursPerWeek',
                            color='Department', title="3D Scatter Plot of Salary, Performance, and Work Hours")
scatter_3d_fig.show()

# Task 6: Subplots for Multiple Charts
fig = make_subplots(rows=1, cols=2, subplot_titles=("Scatter: Salary vs Age", "Line: Salary Growth"))
fig.add_trace(go.Scatter(x=df['Salary'], y=df['Age'], mode='markers', name="Scatter"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['StartDate'], y=df['Salary'], mode='lines', name="Line"), row=1, col=2)
fig.update_layout(title_text="Subplots of Employee Data")
fig.show()


# Task 7: Animation of Time-Series Data
# Create a non-animated scatter plot first
fig = px.scatter(
    df, 
    x="Salary", 
    y="PerformanceRating", 
    color="Department",
    size='Age',
    hover_name="FirstName",
    range_x=[df['Salary'].min()-5000, df['Salary'].max()+5000],
    range_y=[df['PerformanceRating'].min()-0.5, df['PerformanceRating'].max()+0.5],
    title="Employee Performance by Department"
)

# Improve the layout
fig.update_layout(
    xaxis_title="Salary",
    yaxis_title="Performance Rating",
    legend_title="Department",
    height=600,
    width=800
)

# Show the non-animated plot
fig.show()


# Get unique dates for animation frames
dates = df['StartDate'].dt.strftime('%Y-%m-%d').unique()

# Create figure
fig = go.Figure()

# Add traces for each department
for department in df['Department'].unique():
    dept_data = df[df['Department'] == department]
    fig.add_trace(go.Scatter(
        x=dept_data['Salary'],
        y=dept_data['PerformanceRating'],
        mode='markers',
        marker=dict(size=dept_data['Age']),
        name=department,
        text=dept_data['FirstName'],
        hoverinfo='text'
    ))

# Create and add slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Date: "},
    pad={"t": 50},
    steps=[dict(
        method='update',
        args=[{'visible': [(date in df[df['Department'] == dep]['StartDate'].dt.strftime('%Y-%m-%d').values) 
                        for dep in df['Department'].unique()]},
            {'title': f'Employee Performance on {date}'}],
        label=date
    ) for date in dates]
)]

# Update layout
fig.update_layout(
    sliders=sliders,
    title='Animated Scatter Plot of Employee Performance Over Time',
    xaxis_title='Salary',
    yaxis_title='Performance Rating',
    height=600,
    width=1000
)

# Set initial visibility
fig.update_traces(visible=True)

# Show the plot
fig.show()

# Save as HTML
fig.write_html("animated_employee_performance.html")

# Exporting Charts
scatter_fig.write_image("scatter_plot.png")
scatter_fig.write_html("scatter_plot.html")

print("All charts created and exported successfully.")
