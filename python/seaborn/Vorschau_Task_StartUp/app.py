import plotly.express as px

# Sample data
df = px.data.iris()

# Create a scatter plot
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')

# Show the plot
fig.show()