import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the data
employee_data = pd.read_csv('employee_data.csv')
project_data = pd.read_csv('project_data.csv')
financial_data = pd.read_csv('financial_data.csv')
customer_data = pd.read_csv('customer_data.csv')
market_data = pd.read_csv('market_data.csv')


# Data Cleaning and Preparation
def clean_data(df):
    # Remove duplicates
    df.drop_duplicates()
    
     # Ensure numeric columns are cleaned
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].mean())
    
    # Remove outliers (using IQR method)
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

employee_data = clean_data(employee_data)
project_data = clean_data(project_data)
financial_data = clean_data(financial_data)
customer_data = clean_data(customer_data)
market_data = clean_data(market_data)


# Convert date columns to datetime
project_data['StartDate'] = pd.to_datetime(project_data['StartDate'])
financial_data['Date'] = pd.to_datetime(financial_data['Date'])
market_data['Quarter'] = pd.to_datetime(market_data['Quarter'])

# Task 1: Employee Performance and Retention Analysis
def employee_analysis():
    # EDA on employee dataset
    print(employee_data.describe())
    print(employee_data.info())
    
    # Scatter plot of employee performance vs. salary
    plt.figure(figsize=(10, 6))
    plt.scatter(employee_data['Salary'], employee_data['Performance'])
    plt.xlabel('Salary')
    plt.ylabel('Performance')
    plt.title('Employee Performance vs Salary')
    plt.savefig('employee_performance_vs_salary.png')
    plt.close()
    
    # Heatmap of correlation between employee metrics
    corr_matrix = employee_data[['Salary', 'Performance', 'YearsOfExperience', 'SkillLevel']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Employee Metrics')
    plt.savefig('employee_correlation_heatmap.png')
    plt.close()
    
    # Interactive dashboard for HR
    fig = px.scatter(employee_data, x='Salary', y='Performance', color='Department', 
                     hover_data=['Name', 'YearsOfExperience', 'SkillLevel'])
    fig.update_layout(title='Interactive Employee Dashboard')
    fig.write_html('employee_dashboard.html')
    avg_performance = employee_data['Performance'].mean()
    high_performers = employee_data[employee_data['Performance'] > avg_performance + 1]
    low_performers = employee_data[employee_data['Performance'] < avg_performance - 1]

    print("\nEmployee Performance and Retention Analysis Decisions:")
    print(f"1. Consider promotions or bonuses for {len(high_performers)} high-performing employees.")
    print(f"2. Implement performance improvement plans for {len(low_performers)} low-performing employees.")
    print(f"3. Investigate the correlation between salary and performance (correlation: {employee_data['Salary'].corr(employee_data['Performance']):.2f}).")
    if employee_data['Salary'].corr(employee_data['Performance']) < 0.5:
        print("   Consider revising the compensation structure to better align with performance.")

# Task 2: Project Profitability and Timeline Analysis
def project_analysis():
    # Clean and preprocess project data
    project_data['EndDate'] = project_data['StartDate'] + pd.to_timedelta(project_data['Duration'], unit='D')
    project_data['Profitability'] = project_data['Revenue'] - project_data['ActualCost']
    
    # Bar chart of project durations
    plt.figure(figsize=(12, 6))
    plt.bar(project_data['ProjectName'], project_data['Duration'])
    plt.xticks(rotation=90)
    plt.xlabel('Project Name')
    plt.ylabel('Duration (days)')
    plt.title('Project Durations')
    plt.tight_layout()
    plt.savefig('project_durations.png')
    plt.close()
    
    # Scatter plot of project cost vs. revenue
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=project_data, x='ActualCost', y='Revenue', hue='ProjectManager')
    plt.title('Project Cost vs Revenue')
    plt.savefig('project_cost_vs_revenue.png')
    plt.close()
    
    # Interactive Gantt chart
    fig = px.timeline(project_data, x_start='StartDate', x_end='EndDate', y='ProjectName', color='ProjectManager')
    fig.update_layout(title='Project Timeline', xaxis_title='Date', yaxis_title='Project')
    fig.write_html('project_timeline.html')
    avg_profitability = project_data['Profitability'].mean()
    high_profit_projects = project_data[project_data['Profitability'] > avg_profitability + project_data['Profitability'].std()]
    low_profit_projects = project_data[project_data['Profitability'] < avg_profitability - project_data['Profitability'].std()]

    print("\nProject Profitability and Timeline Analysis Decisions:")
    print(f"1. Replicate success factors from {len(high_profit_projects)} highly profitable projects.")
    print(f"2. Review and potentially restructure {len(low_profit_projects)} low-profit projects.")
    print(f"3. Optimize resource allocation for projects with duration > {project_data['Duration'].mean() + project_data['Duration'].std():.0f} days.")

# Task 3: Financial Trend Analysis
def financial_analysis():
    # Time series analysis
    financial_data['Profit'] = financial_data['Revenue'] - financial_data['Expenses']
    
    # Line plot of revenue, expenses, and profit
    plt.figure(figsize=(12, 6))
    plt.plot(financial_data['Date'], financial_data['Revenue'], label='Revenue')
    plt.plot(financial_data['Date'], financial_data['Expenses'], label='Expenses')
    plt.plot(financial_data['Date'], financial_data['Profit'], label='Profit')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('Financial Trends')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('financial_trends.png')
    plt.close()
    
    # Interactive financial dashboard
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=financial_data['Date'], y=financial_data['Revenue'], name='Revenue'))
    fig.add_trace(go.Scatter(x=financial_data['Date'], y=financial_data['Expenses'], name='Expenses'))
    fig.add_trace(go.Scatter(x=financial_data['Date'], y=financial_data['Profit'], name='Profit'))
    fig.update_layout(title='Interactive Financial Dashboard', xaxis_title='Date', yaxis_title='Amount')
    fig.write_html('financial_dashboard.html')
    recent_trend = financial_data['Profit'].tail(3).mean() - financial_data['Profit'].head(3).mean()
    profit_margin = (financial_data['Profit'] / financial_data['Revenue']).mean()

    print("\nFinancial Trend Analysis Decisions:")
    print(f"1. {'Expand operations' if recent_trend > 0 else 'Implement cost-cutting measures'} based on the recent profit trend of ${recent_trend:.2f}.")
    print(f"2. {'Maintain current strategies' if profit_margin > 0.2 else 'Improve profit margins'} (current average: {profit_margin:.2%}).")
    print(f"3. Focus on {'revenue growth' if financial_data['Revenue'].pct_change().mean() < 0.05 else 'cost management'} to improve overall financial performance.")

# Task 4: Customer Satisfaction and Revenue Impact
def customer_analysis():
    # Histogram of customer satisfaction scores
    plt.figure(figsize=(10, 6))
    plt.hist(customer_data['Satisfaction'], bins=20)
    plt.xlabel('Satisfaction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Customer Satisfaction Scores')
    plt.savefig('customer_satisfaction_distribution.png')
    plt.close()
    
    # Pair plot of customer metrics
    sns.pairplot(customer_data[['Satisfaction', 'ContractValue', 'ContractDuration']])
    plt.suptitle('Pair Plot of Customer Metrics', y=1.02)
    plt.savefig('customer_metrics_pairplot.png')
    plt.close()
    
    # Interactive scatter plot of satisfaction vs. revenue
    fig = px.scatter(customer_data, x='Satisfaction', y='ContractValue', 
                     size='ContractDuration', hover_data=['CustomerName'])
    fig.update_layout(title='Customer Satisfaction vs Contract Value')
    fig.write_html('customer_satisfaction_vs_revenue.html')
    # Decision-making suggestions
    avg_satisfaction = customer_data['Satisfaction'].mean()
    low_satisfaction = customer_data[customer_data['Satisfaction'] < avg_satisfaction - 1]
    high_value_low_satisfaction = low_satisfaction[low_satisfaction['ContractValue'] > customer_data['ContractValue'].mean()]

    print("\nCustomer Satisfaction and Revenue Impact Decisions:")
    print(f"1. Implement satisfaction improvement initiatives for {len(low_satisfaction)} customers with below-average satisfaction.")
    print(f"2. Prioritize {len(high_value_low_satisfaction)} high-value customers with low satisfaction for immediate attention.")
    print(f"3. {'Revise pricing strategy' if customer_data['Satisfaction'].corr(customer_data['ContractValue']) < 0 else 'Maintain current pricing'} based on the correlation between satisfaction and contract value.")

# Task 5: Market Share and Competitive Analysis
def market_analysis():
    # Pie chart of market share
    latest_quarter = market_data.iloc[-1]
    plt.figure(figsize=(10, 6))
    plt.pie([latest_quarter['MarketShare'], latest_quarter['CompetitorShare'], 
             1 - latest_quarter['MarketShare'] - latest_quarter['CompetitorShare']], 
            labels=['Galactic Innovations', 'Main Competitor', 'Others'], 
            autopct='%1.1f%%')
    plt.title(f'Market Share (as of {latest_quarter["Quarter"].strftime("%Y-%m")})')
    plt.savefig('market_share_pie.png')
    plt.close()

    # Grouped bar chart of competitor comparison
    market_data_melted = pd.melt(market_data, id_vars=['Quarter'], 
                                 value_vars=['MarketShare', 'CompetitorShare'], 
                                 var_name='Company', value_name='Share')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Quarter', y='Share', hue='Company', data=market_data_melted)
    plt.xticks(rotation=45)
    plt.title('Market Share Comparison')
    plt.tight_layout()
    plt.savefig('market_share_comparison.png')
    plt.close()
    
    # Interactive treemap of market segments
    fig = px.treemap(names=['Total Market', 'Galactic Innovations', 'Main Competitor', 'Others'],
                     parents=['', 'Total Market', 'Total Market', 'Total Market'],
                     values=[100, latest_quarter['MarketShare']*100, 
                             latest_quarter['CompetitorShare']*100, 
                             (1 - latest_quarter['MarketShare'] - latest_quarter['CompetitorShare'])*100])
    fig.update_layout(title=f'Market Segments (as of {latest_quarter["Quarter"].strftime("%Y-%m")})')
    fig.write_html('market_segments_treemap.html')
    market_share_trend = market_data['MarketShare'].pct_change().mean()
    competitor_share_trend = market_data['CompetitorShare'].pct_change().mean()

    print("\nMarket Share and Competitive Analysis Decisions:")
    print(f"1. {'Strengthen market position' if market_share_trend > 0 else 'Implement aggressive growth strategies'} based on the market share trend of {market_share_trend:.2%}.")
    print(f"2. {'Monitor competitor closely' if competitor_share_trend > 0 else 'Capitalize on competitor weakness'} based on the competitor's market share trend of {competitor_share_trend:.2%}.")
    print(f"3. Focus on {'product differentiation' if market_data['MarketShare'].iloc[-1] < 0.3 else 'customer retention'} to maintain competitive advantage.")

# Task 6: Resource Utilization and Optimization
def resource_analysis():
    # For this task, we'll need to create some mock resource allocation data
    departments = ['R&D', 'Engineering', 'Sales', 'HR', 'Finance']
    resources = ['Budget', 'Manpower', 'Equipment']
    resource_data = pd.DataFrame({
        'Department': np.repeat(departments, len(resources)),
        'Resource': resources * len(departments),
        'Utilization': np.random.uniform(0.5, 1, len(departments) * len(resources))
    })
    
    # Stacked bar chart of resource utilization
    plt.figure(figsize=(12, 6))
    resource_data_pivot = resource_data.pivot(index='Department', columns='Resource', values='Utilization')
    resource_data_pivot.plot(kind='bar', stacked=True)
    plt.title('Resource Utilization by Department')
    plt.xlabel('Department')
    plt.ylabel('Utilization')
    plt.legend(title='Resource')
    plt.tight_layout()
    plt.savefig('resource_utilization.png')
    plt.close()
    
    # Heatmap of resource efficiency
    plt.figure(figsize=(10, 8))
    sns.heatmap(resource_data_pivot, annot=True, cmap='YlGnBu')
    plt.title('Resource Efficiency Heatmap')
    plt.savefig('resource_efficiency_heatmap.png')
    plt.close()
    
    # Interactive Sankey diagram of resource flow
    source = [departments.index(dept) for dept in resource_data['Department']]
    target = [len(departments) + resources.index(res) for res in resource_data['Resource']]
    value = resource_data['Utilization'] * 100  # Scale up for better visibility
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = departments + resources,
          color = "blue"
        ),
        link = dict(
          source = source,
          target = target,
          value = value
    ))])
    fig.update_layout(title_text="Resource Flow Sankey Diagram", font_size=10)
    fig.write_html('resource_flow_sankey.html')
    low_utilization = resource_data[resource_data['Utilization'] < 0.7]
    high_utilization = resource_data[resource_data['Utilization'] > 0.9]

    print("\nResource Utilization and Optimization Decisions:")
    print(f"1. Redistribute resources from {len(low_utilization)} underutilized areas to optimize efficiency.")
    print(f"2. Invest in additional resources for {len(high_utilization)} over-utilized areas to prevent burnout and maintain quality.")
    print(f"3. Implement regular resource audits to maintain optimal utilization levels across all departments.")

# Task 7: Technology Innovation Impact Analysis
def innovation_analysis():
    # For this task, we'll need to create some mock technology innovation data
    project_data['InnovationScore'] = np.random.uniform(0, 10, len(project_data))
    project_data['TechnologyComplexity'] = np.random.uniform(1, 10, len(project_data))
    project_data['MarketPotential'] = np.random.uniform(1, 10, len(project_data))
    
    # If 'Department' column doesn't exist, create a mock one
    if 'Department' not in project_data.columns:
        departments = ['R&D', 'Engineering', 'Product', 'Marketing']
        project_data['Department'] = np.random.choice(departments, len(project_data))
    
    # Bubble chart of technology impact
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(project_data['TechnologyComplexity'], project_data['MarketPotential'], 
                s=project_data['InnovationScore']*50, alpha=0.5, c=project_data['InnovationScore'], cmap='viridis')
    plt.colorbar(scatter, label='Innovation Score')
    plt.xlabel('Technology Complexity')
    plt.ylabel('Market Potential')
    plt.title('Technology Impact Analysis')
    for i, txt in enumerate(project_data['ProjectName']):
        plt.annotate(txt, (project_data['TechnologyComplexity'][i], project_data['MarketPotential'][i]))
    plt.savefig('technology_impact_bubble.png')
    plt.close()
    
    # Box plot of innovation scores by department
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Department', y='InnovationScore', data=project_data)
    plt.title('Innovation Scores by Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('innovation_scores_boxplot.png')
    plt.close()
    
    # Interactive 3D scatter plot of technology metrics
    fig = px.scatter_3d(project_data, x='TechnologyComplexity', y='MarketPotential', z='InnovationScore',
                        color='Department', hover_name='ProjectName', size='InnovationScore',
                        labels={'TechnologyComplexity': 'Technology Complexity',
                                'MarketPotential': 'Market Potential',
                                'InnovationScore': 'Innovation Score'})
    fig.update_layout(title='3D Technology Metrics Visualization')
    fig.write_html('technology_metrics_3d.html')

    # Additional analysis: Correlation heatmap
    correlation_matrix = project_data[['InnovationScore', 'TechnologyComplexity', 'MarketPotential']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Innovation Metrics')
    plt.savefig('innovation_correlation_heatmap.png')
    plt.close()

    # Print summary statistics
    print("Summary Statistics of Innovation Metrics:")
    print(project_data[['InnovationScore', 'TechnologyComplexity', 'MarketPotential']].describe())

    # Identify top innovative projects
    top_innovative_projects = project_data.nlargest(5, 'InnovationScore')
    high_impact_projects = project_data[project_data['InnovationScore'] > project_data['InnovationScore'].mean() + project_data['InnovationScore'].std()]
    low_impact_projects = project_data[project_data['InnovationScore'] < project_data['InnovationScore'].mean() - project_data['InnovationScore'].std()]

    print("\nTechnology Innovation Impact Analysis Decisions:")
    print(f"1. Allocate more resources to {len(high_impact_projects)} high-impact innovative projects.")
    print(f"2. Reassess or potentially discontinue {len(low_impact_projects)} low-impact projects.")
    print(f"3. {'Increase R&D budget' if project_data['InnovationScore'].mean() < 7 else 'Maintain current R&D investment'} to drive future innovations.")

# Task 8: Supply Chain and Logistics Analysis
def supply_chain_analysis():
    # For this task, we'll need to create some mock supply chain data
    np.random.seed(42)
    n_suppliers = 50
    supply_data = pd.DataFrame({
        'SupplierID': range(1, n_suppliers + 1),
        'SupplierName': [f'Supplier{i}' for i in range(1, n_suppliers + 1)],
        'DeliveryTime': np.random.normal(10, 2, n_suppliers),  # in days
        'Quality': np.random.uniform(3, 5, n_suppliers),
        'Cost': np.random.uniform(1000, 5000, n_suppliers),
        'Latitude': np.random.uniform(25, 50, n_suppliers),
        'Longitude': np.random.uniform(-130, -70, n_suppliers)
    })
    
    # Line plot of delivery times
    plt.figure(figsize=(12, 6))
    plt.plot(supply_data['SupplierID'], supply_data['DeliveryTime'])
    plt.xlabel('Supplier ID')
    plt.ylabel('Delivery Time (days)')
    plt.title('Supplier Delivery Times')
    plt.tight_layout()
    plt.savefig('supplier_delivery_times.png')
    plt.close()
    
    # Box plot of supplier performance
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Quality', y='Cost', data=supply_data)
    plt.title('Supplier Performance: Quality vs Cost')
    plt.savefig('supplier_performance_boxplot.png')
    plt.close()
    
    # Interactive map of supplier locations
    fig = px.scatter_geo(supply_data, lat='Latitude', lon='Longitude', 
                         hover_name='SupplierName', size='DeliveryTime', 
                         color='Quality', projection='natural earth')
    fig.update_layout(title='Supplier Locations and Performance')
    fig.write_html('supplier_locations_map.html')
    delayed_suppliers = supply_data[supply_data['DeliveryTime'] > supply_data['DeliveryTime'].mean() + supply_data['DeliveryTime'].std()]
    low_quality_suppliers = supply_data[supply_data['Quality'] < supply_data['Quality'].mean() - supply_data['Quality'].std()]

    print("\nSupply Chain and Logistics Analysis Decisions:")
    print(f"1. Renegotiate terms or find alternatives for {len(delayed_suppliers)} suppliers with consistently delayed deliveries.")
    print(f"2. Implement quality improvement programs with {len(low_quality_suppliers)} low-quality suppliers.")
    print(f"3. {'Diversify supplier base' if len(supply_data) < 30 else 'Optimize current supplier relationships'} to manage supply chain risks.")

    
# Task 9: Employee Skill Gap Analysis (continued)
def skill_gap_analysis():
    # For this task, we'll need to create some mock skill data
    skills = ['Python', 'Data Analysis', 'Machine Learning', 'Project Management', 'Communication']
    employee_data['Skills'] = [np.random.choice(skills, size=np.random.randint(1, len(skills)), replace=False).tolist() for _ in range(len(employee_data))]
    
    # Radar chart of skill distribution
    skill_counts = employee_data['Skills'].apply(pd.Series).stack().value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))
    theta = np.linspace(0, 2 * np.pi, len(skill_counts), endpoint=False)
    ax.plot(theta, skill_counts.values)
    ax.set_xticks(theta)
    ax.set_xticklabels(skill_counts.index)
    plt.title('Skill Distribution')
    plt.tight_layout()
    plt.savefig('skill_distribution_radar.png')
    plt.close()
    
    # Clustered heatmap of skill similarities
    skill_matrix = employee_data['Skills'].apply(lambda x: pd.Series([1 if skill in x else 0 for skill in skills]))
    skill_similarity = skill_matrix.T.dot(skill_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.clustermap(skill_similarity, cmap='YlGnBu', annot=True)
    plt.title('Skill Similarities Clustered Heatmap')
    plt.savefig('skill_similarities_heatmap.png')
    plt.close()
    
    # Interactive sunburst chart of skill hierarchies
    skill_hierarchy = {
        'All Skills': {skill: employee_data['Skills'].apply(lambda x: skill in x).sum() for skill in skills}
    }
    
    fig = px.sunburst(
        names=['All Skills'] + list(skill_hierarchy['All Skills'].keys()),
        parents=[''] + ['All Skills'] * len(skill_hierarchy['All Skills']),
        values=[sum(skill_hierarchy['All Skills'].values())] + list(skill_hierarchy['All Skills'].values())
    )
    fig.update_layout(title='Skill Hierarchy Sunburst Chart')
    fig.write_html('skill_hierarchy_sunburst.html')
    most_common_skill = skill_counts.idxmax()
    least_common_skill = skill_counts.idxmin()

    print("\nEmployee Skill Gap Analysis Decisions:")
    print(f"1. Develop training programs to address the skill gap in {least_common_skill}.")
    print(f"2. Leverage expertise in {most_common_skill} for mentorship programs and knowledge sharing.")
    print(f"3. {'Prioritize hiring for' if skill_counts.min() / len(employee_data) < 0.3 else 'Enhance internal training for'} skills with low representation in the workforce.")

# Task 10: Simple Predictive Analysis for Future Projects
def simple_predictive_analysis():
    # For this task, we'll create a simple prediction based on historical data
    # We'll use project budget to predict project duration
    
    # Create a scatter plot of Budget vs Duration
    plt.figure(figsize=(10, 6))
    plt.scatter(project_data['Budget'], project_data['Duration'])
    plt.xlabel('Budget')
    plt.ylabel('Duration')
    plt.title('Project Budget vs Duration')
    plt.savefig('budget_vs_duration_scatter.png')
    plt.close()
    
    # Calculate correlation coefficient
    correlation = project_data['Budget'].corr(project_data['Duration'])
    
    # Create a simple linear model using numpy's polyfit
    coefficients = np.polyfit(project_data['Budget'], project_data['Duration'], 1)
    polynomial = np.poly1d(coefficients)
    
    # Plot the data and the linear model
    plt.figure(figsize=(10, 6))
    plt.scatter(project_data['Budget'], project_data['Duration'], label='Actual Data')
    plt.plot(project_data['Budget'], polynomial(project_data['Budget']), color='red', label='Linear Model')
    plt.xlabel('Budget')
    plt.ylabel('Duration')
    plt.title(f'Project Budget vs Duration (Correlation: {correlation:.2f})')
    plt.legend()
    plt.savefig('budget_vs_duration_model.png')
    plt.close()
    
    # Create an interactive scatter plot with the linear model
    fig = px.scatter(project_data, x='Budget', y='Duration', trendline='ols')
    fig.update_layout(title='Interactive Project Budget vs Duration')
    fig.write_html('interactive_budget_vs_duration.html')

    print(f"Correlation between Budget and Duration: {correlation:.2f}")
    print(f"Linear model: Duration = {coefficients[0]:.2f} * Budget + {coefficients[1]:.2f}")
    print("\nPredictive Analysis for Future Projects Decisions:")
    print(f"1. {'Use the linear model for initial project planning' if correlation > 0.7 else 'Develop more sophisticated prediction models'} based on the correlation between budget and duration.")
    print(f"2. Allocate {'more' if coefficients[0] > 0 else 'less'} time for high-budget projects in future planning.")
    print(f"3. Investigate factors beyond budget that influence project duration to improve prediction accuracy.")

# Main function to run all analyses
def main():
    employee_analysis()
    project_analysis()
    financial_analysis()
    customer_analysis()
    market_analysis()
    resource_analysis()
    innovation_analysis()
    supply_chain_analysis()
    skill_gap_analysis()
    simple_predictive_analysis()

if __name__ == "__main__":
    main()