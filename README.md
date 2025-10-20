# Data-Visualization-using-Python - Matplotlib
Methodology for Data Visualization Using Matplotlib  Data visualization is a crucial step in data analysis, enabling better understanding, pattern recognition, and insight generation. Matplotlib, a powerful Python library, provides a flexible framework for creating static, animated, and interactive visualizations.
Data visualization using Matplotlib follows a structured methodology that encompasses data preparation, iterative plotting, and final refinement to effectively communicate insights.

The methodology for visualizing data using Matplotlibis typically divided into three main phases: Preparation, Visualization, and Refinement.
It typically follows these steps:

1. Importing Required Libraries
The process begins by importing the Matplotlib library, usually its pyplot module, which offers a MATLAB-like interface for plotting:
```import matplotlib.pyplot as plt```

Visualization in a Jupyter Notebook: Matplotlib magic command ensures inline rendering:
```%matplotlib inline```

2. Loading and Inspecting Data
Data can be imported from sources such as CSV, Excel, or SQL databases using libraries like pandas:
```
import pandas as pd
data = pd.read_csv('data.csv')
```
Initial glance of data 
Command,Purpose
df.head(),Displays the first 5 rows of the DataFrame. Crucial for seeing the actual data format.
df.tail(),Displays the last 5 rows. Useful for checking if any end-of-file artifacts or summaries exist.
df.shape,"Returns a tuple (rows, columns), telling you exactly how large the dataset is."
df.columns,Lists the names of all columns (features).

Structure and Data Types
Command,Purpose
df.info(),"Provides a summary of the DataFrame: the number of entries, column names, the count of non-null values for each column, and the data type (dtype) of each column."
df.dtypes,"Lists the data type for every column. Essential for ensuring numerical columns are treated as numbers (e.g., float64 or int64) and categories are objects."

Statistical Summary
Command,Purpose
df.describe(),"Generates descriptive statistics for all numerical columns, including: count, mean, standard deviation, minimum, quartiles (25%, 50%, 75%), and maximum."
df.describe(include='object'),"Generates descriptive statistics for all categorical (object/string) columns, including: count, unique values, top value, and its frequency."

Uniqueness and Specific Values
Command,Purpose
df['column_name'].unique(),Lists all unique values in a specific column.
df['column_name'].nunique(),Returns the count of unique values in a specific column.
df['column_name'].value_counts(),"Returns a Series showing the frequency of each unique value, ordered from most to least frequent. Excellent for checking class imbalance in target variables."
