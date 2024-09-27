import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load the data
employee_df = pd.read_csv('startup_data.csv')
department_df = pd.read_csv('department_data.csv')
company_df = pd.read_csv('company_data.csv')
employee_df.columns = employee_df.columns.str.strip()
department_df.columns = department_df.columns.str.strip()
company_df.columns = company_df.columns.str.strip()
# Merge employee and department data
merged_df = pd.merge(employee_df, department_df, on='Department', how='left')
print(merged_df.columns)

# Function to save plots
def save_plot(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to save the plot
def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

# Function that handles plot generation
def generate_plot(plot_type, x=None, y=None, hue=None, data=None, title='', filename='', **kwargs):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    match plot_type:
        case 'boxplot':
            sns.boxplot(x=x, y=y, data=data, ax=ax, **kwargs)
        case 'scatterplot':
            sns.scatterplot(x=x, y=y, hue=hue, data=data, ax=ax, **kwargs)
        case 'regplot':
            sns.regplot(x=x, y=y, data=data, ax=ax, **kwargs)
        case _:
            print(f"Plot type '{plot_type}' is not recognized.")
            return
    
    ax.set_title(title)
    if x and y:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    
    # Rotate x-axis labels if needed
    if plot_type == 'boxplot':
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Save and close the plot
    save_plot(fig, filename)

# Salary Analysis Function
def salary_analysis(merged_df):
    # a. Salary distribution across departments
    generate_plot(
        plot_type='boxplot',
        x='Department',
        y='Salary',
        data=merged_df,
        title='Salary Distribution Across Departments',
        filename='1a_salary_distribution.png'
    )

    # b. Salary vs. years of experience
    generate_plot(
        plot_type='scatterplot',
        x='YearsOfExperience',
        y='Salary',
        hue='Department',
        data=merged_df,
        title='Salary vs Years of Experience by Department',
        filename='1b_salary_vs_experience.png',
        figsize=(12, 8)
    )

    # c. Correlation between performance rating and salary
    correlation = merged_df['Salary'].corr(merged_df['PerformanceRating'])
    generate_plot(
        plot_type='regplot',
        x='PerformanceRating',
        y='Salary',
        data=merged_df,
        title=f'Correlation between Salary and Performance Rating: {correlation:.2f}',
        filename='1c_salary_performance_correlation.png',
        figsize=(10, 6)
    )


# 1. Salary Analysis
def salary_analysis():
    # a. Salary distribution across departments
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Department', y='Salary', data=merged_df, ax=ax)
    ax.set_title('Salary Distribution Across Departments')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '1a_salary_distribution.png')

    # b. Salary vs. years of experience
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='YearsOfExperience', y='Salary', hue='Department', data=merged_df, ax=ax)
    ax.set_title('Salary vs Years of Experience by Department')
    save_plot(fig, '1b_salary_vs_experience.png')

    # c. Correlation between performance rating and salary
    correlation = merged_df['Salary'].corr(merged_df['PerformanceRating'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='PerformanceRating', y='Salary', data=merged_df, ax=ax)
    ax.set_title(f'Correlation between Salary and Performance Rating: {correlation:.2f}')
    save_plot(fig, '1c_salary_performance_correlation.png')

# 2. Performance Evaluation
def performance_evaluation():
    # a. Heatmap of correlations
    corr_data = merged_df[['PerformanceRating', 'YearsOfExperience', 'ClientSatisfactionScore']]
    corr_matrix = corr_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    save_plot(fig, '2a_correlation_heatmap.png')

    # b. Average performance ratings by department
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Department', y='PerformanceRating', data=merged_df, ax=ax)
    ax.set_title('Average Performance Ratings by Department')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '2b_avg_performance_by_dept.png')

    # c. Training hours vs. performance rating
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='TrainingHours', y='PerformanceRating', data=merged_df, ax=ax)
    ax.set_title('Training Hours vs Performance Rating')
    save_plot(fig, '2c_training_vs_performance.png')

# 3. Employee Retention
def employee_retention():
    # a. Average tenure by department
    merged_df['Tenure'] = (pd.Timestamp.now() - pd.to_datetime(merged_df['StartDate'])).dt.days / 365
    avg_tenure = merged_df.groupby('Department')['Tenure'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_tenure.plot(kind='bar', ax=ax)
    ax.set_title('Average Tenure by Department')
    ax.set_ylabel('Years')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '3a_avg_tenure_by_dept.png')

    # b. Vacation days vs. work hours
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(merged_df['WorkHoursPerWeek'], merged_df['VacationDaysTaken'], 
        c=merged_df['PerformanceRating'], cmap='viridis')
    ax.set_xlabel('Work Hours Per Week')
    ax.set_ylabel('Vacation Days Taken')
    ax.set_title('Vacation Days vs Work Hours')
    plt.colorbar(scatter, label='Performance Rating')
    save_plot(fig, '3b_vacation_vs_work_hours.png')

    # c. Age distribution across roles
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='Role', y='Age', data=merged_df, ax=ax)
    ax.set_title('Age Distribution Across Roles')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '3c_age_distribution_roles.png')

# 4. Project Analysis
def project_analysis():
    # a. Average team size by project
    avg_team_size = merged_df.groupby('Project')['TeamSize'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_team_size.plot(kind='bar', ax=ax)
    ax.set_title('Average Team Size by Project')
    ax.set_ylabel('Average Team Size')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '4a_avg_team_size_by_project.png')

    # b. Performance rating and client satisfaction by project
    project_metrics = merged_df.groupby('Project').agg({
        'PerformanceRating': 'mean',
        'ClientSatisfactionScore': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(project_metrics))
    width = 0.35
    ax.bar(x, project_metrics['PerformanceRating'], width, label='Performance Rating')
    ax.bar([i + width for i in x], project_metrics['ClientSatisfactionScore'], width, label='Client Satisfaction')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(project_metrics['Project'], rotation=45, ha='right')
    ax.legend()
    ax.set_title('Performance Rating and Client Satisfaction by Project')
    save_plot(fig, '4b_performance_satisfaction_by_project.png')

    # c. Salary distribution across projects
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Project', y='Salary', data=merged_df, ax=ax)
    ax.set_title('Salary Distribution Across Projects')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '4c_salary_distribution_projects.png')

# 5. Compensation Structure
def compensation_structure():
    # a. Total compensation distribution
    merged_df['TotalCompensation'] = merged_df['Salary'] + merged_df['Bonuses']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(merged_df['TotalCompensation'], kde=True, ax=ax)
    ax.set_title('Distribution of Total Compensation')
    ax.set_xlabel('Total Compensation')
    save_plot(fig, '5a_total_compensation_distribution.png')

    # b. Total compensation vs. performance rating
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(merged_df['PerformanceRating'], merged_df['TotalCompensation'], 
                s=merged_df['YearsOfExperience']*10, alpha=0.6)
    ax.set_xlabel('Performance Rating')
    ax.set_ylabel('Total Compensation')
    ax.set_title('Total Compensation vs Performance Rating')
    plt.colorbar(scatter, label='Years of Experience')
    save_plot(fig, '5b_compensation_vs_performance.png')

    # c. Education level vs. total compensation
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Education', y='TotalCompensation', data=merged_df, ax=ax)
    ax.set_title('Total Compensation by Education Level')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '5c_compensation_by_education.png')

# 6. Workload and Productivity
def workload_productivity():
    # a. Work hours vs. performance rating
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='WorkHoursPerWeek', y='PerformanceRating', hue='Department', data=merged_df, ax=ax)
    ax.set_title('Work Hours vs Performance Rating')
    save_plot(fig, '6a_work_hours_vs_performance.png')

    # b. Training hours vs. client satisfaction
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='TrainingHours', y='ClientSatisfactionScore', data=merged_df, ax=ax)
    ax.set_title('Training Hours vs Client Satisfaction Score')
    save_plot(fig, '6b_training_vs_satisfaction.png')

    # c. Average vacation days by department
    avg_vacation = merged_df.groupby('Department')['VacationDaysTaken'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_vacation.plot(kind='bar', ax=ax)
    ax.set_title('Average Vacation Days Taken by Department')
    ax.set_ylabel('Average Vacation Days')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '6c_avg_vacation_by_dept.png')

# 7. Department Comparisons
def department_comparisons():
    # a. Education levels within departments
    education_dept = pd.crosstab(merged_df['Department'], merged_df['Education'], normalize='index')
    fig, ax = plt.subplots(figsize=(12, 6))
    education_dept.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Education Levels within Departments')
    ax.legend(title='Education', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '7a_education_levels_by_dept.png')

    # b. Radar chart of department metrics
    dept_metrics = merged_df.groupby('Department').agg({
        'Salary': 'mean',
        'PerformanceRating': 'mean',
        'ClientSatisfactionScore': 'mean'
    })
    dept_metrics_normalized = (dept_metrics - dept_metrics.min()) / (dept_metrics.max() - dept_metrics.min())
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    for dept in dept_metrics_normalized.index:
        values = dept_metrics_normalized.loc[dept].values
        values = np.concatenate((values, [values[0]]))  # repeat the first value to close the polygon
        angles = np.linspace(0, 2*np.pi, len(dept_metrics_normalized.columns)+1, endpoint=True)
        ax.plot(angles, values, 'o-', linewidth=2, label=dept)
        ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, dept_metrics_normalized.columns)
    ax.set_title('Department Comparison')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    save_plot(fig, '7b_dept_comparison_radar.png')

    # c. Age distribution across departments
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='Department', y='Age', data=merged_df, ax=ax)
    ax.set_title('Age Distribution Across Departments')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '7c_age_distribution_depts.png')

# 8. Efficiency Metrics
def efficiency_metrics():
    # a. Efficiency score distribution
    merged_df['EfficiencyScore'] = merged_df['PerformanceRating'] / merged_df['WorkHoursPerWeek']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Department', y='EfficiencyScore', data=merged_df, ax=ax)
    ax.set_title('Efficiency Score Distribution Across Departments')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_plot(fig, '8a_efficiency_score_distribution.png')

    # b. Efficiency score vs. total compensation
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='TotalCompensation', y='EfficiencyScore', hue='Role', data=merged_df, ax=ax)
    ax.set_title('Efficiency Score vs Total Compensation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, '8b_efficiency_vs_compensation.png')

    # c. Team size vs. average performance rating
    project_metrics = merged_df.groupby('Project').agg({
        'TeamSize': 'mean',
        'PerformanceRating': 'mean'
    }).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='TeamSize', y='PerformanceRating', data=project_metrics, ax=ax)
    for i, row in project_metrics.iterrows():
        ax.annotate(row['Project'], (row['TeamSize'], row['PerformanceRating']))
    ax.set_title('Team Size vs Average Performance Rating by Project')
    save_plot(fig, '8c_team_size_vs_performance.png')

salary_analysis()
performance_evaluation()
employee_retention()
project_analysis()
compensation_structure()
workload_productivity()
department_comparisons()
efficiency_metrics()