# Data-Visualization-using-Python

Methodology for Data VisualizationMatplotlib and Seaborn are the premier Python tools for data visualization, a crucial process that drives insight, pattern recognition, and deeper understanding in data analysis. While both follow a three-phase methodology, their implementation differs.

Phase 1: Data Preparation and Understanding

This initial phase is identical for both libraries and is necessary to ensure data quality and clarity.
Define the Goal: What specific relationships or distributions are you trying to communicate?
Load and Inspect Data (Pandas): Use df.head(), df.info(), and df.describe() to check structure, data types, and potential issues (like missing values).
Data Transformation: Aggregate, filter, or clean the data to prepare it for the specific chart type (e.g., calculating means for a Bar/Radar chart).

Phase 2: Visualization and Exploration
This phase is where the approaches diverge. Matplotlib requires more steps to build the visual elements, while Seaborn handles most of the statistical mapping internally.
| Library | Primary Focus | Typical Syntax | Key Methodology |
| --- | --- | --- | --- |
| Matplotlib | Flexibility and Customization | ax.plot(), ax.scatter() | Explicit Plotting: You manually map data arrays (X, Y) to visual elements (color, marker) on a manually created axes object. |
| Seaborn | Statistical Graphics & Aesthetics | sns.scatterplot(), sns.histplot() | Declarative Mapping: You tell Seaborn the DataFrame (data=df), the variable for the x-axis (x='feature'), and the grouping variable (hue='category'). Seaborn handles aggregation and styling automatically. |

Phase 3: Refinement and Communication
This final phase relies heavily on Matplotlib's foundational layer for fine-tuning.
| Step | Purpose| Matplotlib/Seaborn Tool |
| Labeling | Add titles, axis labels, and legends. | plt.title(), ax.set_xlabel(), ax.legend() |
|Aesthetics | Adjust colors, line styles, and grids. | Matplotlib commands (ax.grid(), ax.tick_params()) or Seaborn color palettes (palette=...). |
| Clarity | Prevent overlapping elements. | plt.tight_layout() |
| Saving | Finalize the image for presentation. | plt.savefig('output.png', dpi=300) |

Following Steps are essential in data visualization using python

1. Importing Required Libraries
The process begins by importing the Matplotlib library, usually its pyplot module, which offers a MATLAB-like interface for plotting:
```
import matplotlib.pyplot as plt
```

Visualization in a Jupyter Notebook: Matplotlib magic command ensures inline rendering:
```
%matplotlib inline
```

2. Loading and Inspecting Data
Data can be imported from sources such as CSV, Excel, or SQL databases using libraries like pandas:
```
import pandas as pd
data = pd.read_csv('data.csv')
```
Initial glance of data 

| Command | Purpose |
| --- | --- |
| df.head() | Displays the first 5 rows of the DataFrame. Crucial for seeing the actual data format. |
| df.tail() | Displays the last 5 rows. Useful for checking if any end-of-file artifacts or summaries exist. |
| df.shape | Returns a tuple (rows, columns), telling you exactly how large the dataset is. |
| df.columns | Lists the names of all columns (features). |

Structure and Data Types

| Command | Purpose |
| --- | --- |
| df.info() | Provides a summary of the DataFrame: the number of entries, column names, the count of non-null values for each column, and the data type (dtype) of each column. |
| df.dtypes | Lists the data type for every column. Essential for ensuring numerical columns are treated as numbers (e.g., float64 or int64) and categories are objects. |

Statistical Summary

| Command | Purpose |
| --- | --- |
| df.describe() | Generates descriptive statistics for all numerical columns, including: count, mean, standard deviation, minimum, quartiles (25%, 50%, 75%), and maximum. |
| df.describe(include='object') | "Generates descriptive statistics for all categorical (object/string) columns, including: count, unique values, top value, and its frequency. |

Uniqueness and Specific Values

| Command | Purpose |
| --- | --- |
| df['column_name'].unique() | Lists all unique values in a specific column. |
| df['column_name'].nunique() | Returns the count of unique values in a specific column. |
| df['column_name'].value_counts() | Returns a Series showing the frequency of each unique value, ordered from most to least frequent. Excellent for checking class imbalance in target variables. |

3. Selecting the Appropriate visualization charts depends on your objective, data type, and relationship you want to highlight.

| Visualization Type	| Description |	Typical Use Case |
| --- | --- | --- |
| Line Plot	| Displays data trends or changes over a continuous interval or time period.	| Tracking metrics over time (e.g., production, temperature). |
| Bar Chart	| Represents categorical data with rectangular bars proportional to their values. |	Comparing quantities across different categories. |
| Histogram	| Shows the frequency distribution of numerical data divided into bins. |	Understanding distribution and skewness of continuous variables. |
| Pie Diagram	| Circular chart divided into slices to illustrate numerical proportions.	| Showing parts of a whole (e.g., market share). |
| Donut Diagram	| Similar to a pie chart but with a central hole, enhancing readability. |	Emphasizing proportions while allowing space for central labels. |
| Scatter Plot	| Displays relationship between two numerical variables using dots.	| Detecting correlation or clustering patterns. |
| Box Plot	| Summarizes distribution using median, quartiles, and potential outliers.	| Comparing spread and detecting outliers across groups. |
| Violin Plot	| Combines box plot with a density plot to show data distribution and spread.	| Understanding both summary statistics and data distribution shape. |
| Bubble Plot	| Extension of scatter plot where bubble size represents an additional variable.	| Visualizing three variables simultaneously (x, y, and bubble size). |
| Radar Plot (Spider Chart)	| Plots multiple quantitative variables on axes starting from a common center.	| Comparing feature profiles across categories (e.g., species characteristics). |
| Correlation Heatmap	| Visualizes pairwise correlation coefficients between variables using color gradients. |	Identifying strength and direction of relationships between variables. |


| Plot Type | Matplotlib Function | Purpose |
| --- | --- | --- |
| Relationship | plt.scatter() | Show correlation or cluster separation (e.g., Iris data).
| Distribution | plt.hist()\ plt.boxplot()\ sns.violinplot() | Show the frequency, spread, and central tendency of a single variable.|
| Comparison | plt.bar()\ plt.plot() | Compare quantities across different categories or track trends over time.|

```
plt.scatter(data['x'], data['y'])
plt.title('Relationship between X and Y')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.show()
```

4. Customizing the Visualization

Matplotlib allows customization of almost every visual element to improve readability:

Titles and labels for context

Legends for category identification

Color maps and styles for aesthetics

Gridlines and annotations for emphasis

```
plt.plot(data['Year'], data['Sales'], color='teal', linestyle='--', marker='o')
plt.title('Annual Sales Trend')
plt.xlabel('Year')
plt.ylabel('Sales (in USD)')
plt.grid(True)
plt.show()
```

5. Combining Multiple Plots

Subplots help compare multiple visualizations side by side for deeper analysis:

```
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(data['Revenue'], bins=20, color='skyblue')
ax[1].boxplot(data['Revenue'])
plt.show()
```

6. Interpreting and Communicating Insights

The final step involves interpreting patterns, anomalies, and relationships revealed by the plots. Clear labeling, color consistency, and concise legends help communicate insights effectively.

7. Saving and Sharing Visualizations

Plots can be exported in multiple formats for reports or dashboards:
```
plt.savefig('visualization.png', dpi=300, bbox_inches='tight')
```

Included with the repository is Visualization of Iris dataset using Matplotlib. 
