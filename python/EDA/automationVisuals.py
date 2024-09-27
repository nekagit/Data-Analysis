import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import matplotlib.image as mpimg


# AUTOMATIC SEABORN PLOTTING FUNCTION 
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


# AUTOMATIC MATPLOTLIB FUNCTION

# Helper function to save the plot
def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

import matplotlib.pyplot as plt

# Function that handles Matplotlib plot generation
def generate_matplotlib_plot(plot_type, x=None, y=None, data=None, title='', xlabel='', ylabel='', filename='', **kwargs):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    match plot_type:
        case 'line':
            ax.plot(data[x], data[y], **kwargs)
        case 'scatter':
            ax.scatter(data[x], data[y], **kwargs)
        case 'bar':
            ax.bar(data[x], data[y], **kwargs)
        case 'hist':
            ax.hist(data[x], **kwargs)
        case 'box':
            ax.boxplot(data[y], **kwargs)
        case 'area':
            ax.fill_between(data[x], data[y], **kwargs)
        case 'pie':
            ax.pie(data[y], labels=data[x], **kwargs)
        case 'heatmap':
            cax = ax.imshow(data, **kwargs)
            fig.colorbar(cax, ax=ax)
        case 'contour':
            ax.contour(data[x], data[y], **kwargs)
        case '3d':
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(data[x], data[y], data['z'], **kwargs)
        case 'violin':
            ax.violinplot(data[y], **kwargs)
        case 'hexbin':
            ax.hexbin(data[x], data[y], gridsize=50, **kwargs)
        case 'step':
            ax.step(data[x], data[y], **kwargs)
        case 'stem':
            ax.stem(data[x], data[y], **kwargs)
        case 'bubble':
            ax.scatter(data[x], data[y], s=data['size'], **kwargs)
        case 'dendrogram':
            from scipy.cluster.hierarchy import dendrogram
            dendrogram(data, ax=ax, **kwargs)
        case 'pair':
            from pandas.plotting import scatter_matrix
            scatter_matrix(data, ax=ax, **kwargs)
        case 'wordcloud':
            from wordcloud import WordCloud
            wordcloud = WordCloud().generate(' '.join(data))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
        case _:
            print(f"Plot type '{plot_type}' is not recognized.")
            return
    
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)

    # Rotate x-axis labels if needed
    if plot_type == 'bar':
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Save and close the plot
    save_plot(fig, filename)

# Function to save the plot
def save_plot(fig, filename):
    if filename:
        fig.savefig(filename)
    plt.close(fig)

# Salary Analysis Function using Matplotlib
def salary_analysis_matplotlib(merged_df):
    # a. Salary distribution across departments (using bar plot as an example)
    generate_matplotlib_plot(
        plot_type='bar',
        x='Department',
        y='Salary',
        data=merged_df,
        title='Salary Distribution Across Departments (Bar Plot)',
        xlabel='Department',
        ylabel='Salary',
        filename='1a_salary_distribution_matplotlib.png'
    )

    # b. Salary vs. years of experience (using scatter plot)
    generate_matplotlib_plot(
        plot_type='scatter',
        x='YearsOfExperience',
        y='Salary',
        data=merged_df,
        title='Salary vs Years of Experience (Scatter Plot)',
        xlabel='Years of Experience',
        ylabel='Salary',
        filename='1b_salary_vs_experience_matplotlib.png'
    )

    # c. Correlation between performance rating and salary (using line plot as an example)
    correlation = merged_df['Salary'].corr(merged_df['PerformanceRating'])
    generate_matplotlib_plot(
        plot_type='line',
        x='PerformanceRating',
        y='Salary',
        data=merged_df,
        title=f'Salary and Performance Rating Correlation: {correlation:.2f} (Line Plot)',
        xlabel='Performance Rating',
        ylabel='Salary',
        filename='1c_salary_performance_correlation_matplotlib.png'
    )


