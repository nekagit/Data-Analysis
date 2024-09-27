import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import matplotlib.image as mpimg

# Ensure folder structure for each analysis task
def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load the datasets
employee_data = pd.read_csv('employee_data_with_anomalies.csv')
project_data = pd.read_csv('project_data_with_anomalies.csv')
financial_data = pd.read_csv('financial_data_with_anomalies.csv')
customer_data = pd.read_csv('customer_data_with_anomalies.csv')
market_data = pd.read_csv('market_data_with_anomalies.csv')

employee_data.columns = employee_data.columns.str.strip() 
project_data.columns = project_data.columns.str.strip() 
financial_data.columns = financial_data.columns.str.strip() 
customer_data.columns = customer_data.columns.str.strip() 
market_data.columns = market_data.columns.str.strip() 

# Data Cleaning and Preparation
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates().copy()  # Ensure you're working with a new DataFrame copy
    
    # Ensure numeric columns are cleaned
    for column in df.select_dtypes(include=[np.number]).columns:
        df.loc[:, column] = df[column].fillna(df[column].mean())  # Use .loc for setting values
        
        # Clear negative values
        df.loc[:, column] = df[column].clip(lower=0)  # Use .loc for setting values
    
    # Remove outliers (using IQR method)
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()  # Use .copy() to prevent chained indexing warnings
    
    return df

employee_data = clean_data(employee_data)
project_data = clean_data(project_data)
financial_data = clean_data(financial_data)
customer_data = clean_data(customer_data)
market_data = clean_data(market_data)

# Task 1: Employee Performance and Retention Analysis
def employee_analysis():
    
    create_folder('plots')
    plt.figure(figsize=(10, 6))
    plt.scatter(employee_data['Salary'], employee_data['Performance'])
    plt.xlabel('Salary')
    plt.ylabel('Performance')
    plt.title('Employee Performance vs Salary')
    plt.savefig('plots/employee_performance_vs_salary.png')
    plt.close()

    correlation_matrix = employee_data[['Salary', 'Performance', 'YearsOfExperience', 'SkillLevel']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Employee Metrics')
    plt.savefig('plots/employee_correlation_heatmap.png')
    plt.close()

# Task 2: Project Profitability and Timeline Analysis
def project_analysis():
    
    project_data['EndDate'] = pd.to_datetime(project_data['StartDate']) + pd.to_timedelta(project_data['Duration'], unit='D')
    project_data['Profitability'] = project_data['Revenue'] - project_data['ActualCost']

    plt.figure(figsize=(12, 6))
    plt.bar(project_data['ProjectName'], project_data['Duration'])
    plt.xticks(rotation=90)
    plt.xlabel('Project Name')
    plt.ylabel('Duration (days)')
    plt.title('Project Durations')
    plt.tight_layout()
    plt.savefig('plots/project_durations.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=project_data, x='ActualCost', y='Revenue', hue='Profitability', size='Duration', sizes=(20, 200))
    plt.title('Project Cost vs Revenue')
    plt.savefig('plots/project_cost_vs_revenue.png')
    plt.close()

# Task 3: Financial Trend Analysis
def financial_analysis():
    
    financial_data['Date'] = pd.to_datetime(financial_data['Date'])
    financial_data.set_index('Date', inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(financial_data.index, financial_data['Revenue'], label='Revenue')
    plt.plot(financial_data.index, financial_data['Expenses'], label='Expenses')
    plt.plot(financial_data.index, financial_data['Profit'], label='Profit')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('Financial Trends Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/financial_trends.png')
    plt.close()

# Task 4: Customer Satisfaction and Revenue Impact
def customer_analysis():
    
    plt.figure(figsize=(10, 6))
    plt.hist(customer_data['Satisfaction'], bins=20)
    plt.xlabel('Satisfaction Score')
    plt.ylabel('Count')
    plt.title('Distribution of Customer Satisfaction Scores')
    plt.savefig('plots/customer_satisfaction_distribution.png')
    plt.close()

    sns.pairplot(customer_data[['Satisfaction', 'ContractValue', 'ContractDuration']])
    plt.suptitle('Pair Plot of Customer Metrics', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/customer_metrics_pairplot.png')
    plt.close()

# Task 5: Market Share and Competitive Analysis
def market_analysis():
    
    latest_quarter = market_data['Quarter'].max()
    latest_data = market_data[market_data['Quarter'] == latest_quarter]

    plt.figure(figsize=(10, 6))
    plt.pie([latest_data['MarketShare'].iloc[0], latest_data['CompetitorShare'].iloc[0], 
             1 - latest_data['MarketShare'].iloc[0] - latest_data['CompetitorShare'].iloc[0]], 
            labels=['Company', 'Competitors', 'Others'], autopct='%1.1f%%')
    plt.title(f'Market Share Distribution (as of {latest_quarter})')
    plt.savefig('plots/market_share_pie_chart.png')
    plt.close()

# Task 6: Resource Utilization and Optimization
def resource_analysis():
    
    # For this task, we'll need to create some dummy resource allocation data
    resources = ['R&D', 'Engineering', 'Sales', 'HR', 'Finance']
    projects = project_data['ProjectName'].unique()[:5]  # Take first 5 projects for simplicity
    resource_allocation = pd.DataFrame(np.random.randint(1, 10, size=(len(resources), len(projects))), 
                                       index=resources, columns=projects)

    plt.figure(figsize=(12, 6))
    resource_allocation.plot(kind='bar', stacked=True)
    plt.title('Resource Allocation Across Projects')
    plt.xlabel('Resources')
    plt.ylabel('Allocation')
    plt.legend(title='Projects', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/resource_allocation.png')
    plt.close()

# Task 7: Technology Innovation Impact Analysis
def innovation_analysis():
    
    # For this task, we'll need to create some dummy technology impact data
    tech_impact = pd.DataFrame({
        'Technology': ['Tech A', 'Tech B', 'Tech C', 'Tech D', 'Tech E'],
        'Impact': np.random.uniform(0, 1, 5),
        'Adoption': np.random.uniform(0, 1, 5),
        'Cost': np.random.uniform(100000, 1000000, 5)
    })

    plt.figure(figsize=(10, 6))
    plt.scatter(tech_impact['Adoption'], tech_impact['Impact'], s=tech_impact['Cost']/10000, alpha=0.5)
    for i, txt in enumerate(tech_impact['Technology']):
        plt.annotate(txt, (tech_impact['Adoption'][i], tech_impact['Impact'][i]))
    plt.xlabel('Adoption Rate')
    plt.ylabel('Impact')
    plt.title('Technology Impact Analysis')
    plt.savefig('plots/technology_impact.png')
    plt.close()

# Task 8: Supply Chain and Logistics Analysis
def supply_chain_analysis():
    create_folder('plots')
    
    # For this task, we'll need to create some dummy supply chain data
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
    delivery_times = pd.DataFrame({
        'Supplier': suppliers,
        'AverageDeliveryTime': np.random.uniform(1, 10, 5),
        'DeliveryVariance': np.random.uniform(0, 2, 5)
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Supplier', y='AverageDeliveryTime', data=delivery_times)
    plt.title('Supplier Delivery Time Performance')
    plt.ylabel('Average Delivery Time (days)')
    plt.savefig('plots/supplier_delivery_performance.png')
    plt.close()

# Task 9: Employee Skill Gap Analysis
def skill_gap_analysis():
    
    skills = ['Technical', 'Management', 'Communication', 'Problem Solving', 'Teamwork']
    current_skills = np.random.uniform(0, 10, 5)
    required_skills = np.random.uniform(5, 10, 5)

    plt.figure(figsize=(10, 6))
    angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    current_skills = np.concatenate((current_skills, [current_skills[0]]))
    required_skills = np.concatenate((required_skills, [required_skills[0]]))

    plt.polar(angles, current_skills, 'o-', linewidth=2, label='Current Skills')
    plt.polar(angles, required_skills, 'o-', linewidth=2, label='Required Skills')
    plt.fill(angles, current_skills, alpha=0.25)
    plt.fill(angles, required_skills, alpha=0.25)

    plt.xticks(angles[:-1], skills)
    plt.title('Skill Gap Analysis')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('plots/skill_gap_analysis.png')
    plt.close()
    
def combine_plots():
    # List of image file paths (add paths for all your images)
    image_paths = [
        'plots/employee_performance_vs_salary.png',
        'plots/employee_correlation_heatmap.png',
        'plots/project_durations.png',
        'plots/project_cost_vs_revenue.png',
        'plots/financial_trends.png',
        'plots/customer_satisfaction_distribution.png',
        'plots/customer_metrics_pairplot.png',
        'plots/market_share_pie_chart.png',
        'plots/resource_allocation.png',
        'plots/technology_impact.png',
        'plots/supplier_delivery_performance.png',
        'plots/skill_gap_analysis.png'
    ]
    
    # Number of rows and columns for the subplots (adjust grid size according to number of plots)
    n_rows = 4
    n_cols = 3

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Loop through image paths and axes to load and plot images
    for i, img_path in enumerate(image_paths):
        img = mpimg.imread(img_path)  # Load image
        axes[i].imshow(img)           # Display image in the subplot
        axes[i].axis('off')           # Hide axes for a cleaner look

    # Adjust layout
    plt.tight_layout()

    # Save the combined figure
    plt.savefig('plots/combined_plots.png')
    

# Run all analyses
employee_analysis()
project_analysis()
financial_analysis()
customer_analysis()
market_analysis()
resource_analysis()
innovation_analysis()
supply_chain_analysis()
skill_gap_analysis()

# Call the function to combine and display the plots
combine_plots()

# Generate Comprehensive Company Health Report
def generate_report():
    report = """
    # Comprehensive Company Health Report for Galactic Innovations Inc.

    ## 1. Employee Performance and Retention
    - Average Salary: ${:,.2f}
    - Average Performance Score: {:.2f}
    - Correlation between Salary and Performance: {:.2f}

    ## 2. Project Profitability and Timeline
    - Average Project Duration: {:.2f} days
    - Average Project Profitability: ${:,.2f}
    - Most Profitable Project: {}

    ## 3. Financial Trends
    - Average Monthly Revenue: ${:,.2f}
    - Average Monthly Expenses: ${:,.2f}
    - Average Monthly Profit: ${:,.2f}

    ## 4. Customer Satisfaction and Revenue Impact
    - Average Customer Satisfaction: {:.2f}
    - Correlation between Satisfaction and Contract Value: {:.2f}

    ## 5. Market Share and Competitive Analysis
    - Current Market Share: {:.2f}%
    - Competitor Market Share: {:.2f}%

    ## 6. Resource Utilization and Optimization
    - Most Utilized Department: {}
    - Least Utilized Department: {}

    ## 7. Technology Innovation Impact
    - Most Impactful Technology: {}
    - Technology with Highest Adoption Rate: {}

    ## 8. Supply Chain and Logistics
    - Best Performing Supplier: {}
    - Worst Performing Supplier: {}

    ## 9. Employee Skill Gap
    - Largest Skill Gap: {}
    - Smallest Skill Gap: {}

    Please refer to the generated visualizations for more detailed insights.
    """.format(
        employee_data['Salary'].mean(),
        employee_data['Performance'].mean(),
        employee_data['Salary'].corr(employee_data['Performance']),
        project_data['Duration'].mean(),
        project_data['Profitability'].mean(),
        project_data.loc[project_data['Profitability'].idxmax(), 'ProjectName'],
        financial_data['Revenue'].mean(),
        financial_data['Expenses'].mean(),
        financial_data['Profit'].mean(),
        customer_data['Satisfaction'].mean(),
        customer_data['Satisfaction'].corr(customer_data['ContractValue']),
        market_data.iloc[-1]['MarketShare'] * 100,
        market_data.iloc[-1]['CompetitorShare'] * 100,
        employee_data['Department'].value_counts().index[0],
        employee_data['Department'].value_counts().index[-1],
        'Tech A',  # Placeholder
        'Tech B',  # Placeholder
        'Supplier A',  # Placeholder
        'Supplier E',  # Placeholder
        'Technical',  # Placeholder
        'Teamwork'  # Placeholder
    )

    with open('company_health_report.md', 'w') as f:
        f.write(report)

generate_report()

print("Analysis complete. All visualizations and the comprehensive report have been generated.")