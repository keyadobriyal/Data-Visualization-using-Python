# The following code is correct and ready to run, assuming the libraries are installed.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
# %matplotlib inline                          # only in Jupyter

# Load the dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target (species) as a column. The target is an array of 0s, 1s, and 2s.
# We map these to the actual species names (Setosa, Versicolor, Virginica).
df['species'] = pd.Series(iris.target).apply(lambda x: iris.target_names[x])

# Display the first few rows to verify
print("First 5 Rows:")
print(df.head(), "\n")
print("Dataset Shape:")
print(df.shape, "\n")

# --- Data Inspection ---
#Display data type
print("Dataset Info:")
print(df.info(), "\n")

# --- Data Statistics ---
#Display Summary Statistics
print("Summary Statistics:")
print(df.describe(), "\n")

#Check for missing values
print("Missing Values:")
print(df.isnull().sum(), "\n")

#Display Unique Species
print("Unique Species:")
print(df['species'].unique(), "\n")

# Filter dataframes for each species
setosa = df[df['species'] == 'setosa']
versicolor = df[df['species'] == 'versicolor']
virginica = df[df['species'] == 'virginica']


# --- Histograms ---
# Prepare Histogram of Petal Length
plt.figure(figsize=(7,4))
# Note: Using df['petal length (cm)'] is safer than iris['petal_length'] as 'iris' is the bunch object
plt.hist(df['petal length (cm)'], bins=20, edgecolor='black') 
plt.title('Distribution of Petal Length')
plt.xlabel('Petal length (cm)')
plt.ylabel('Count')
plt.show()

# Prepare Histogram of Sepal Length of all three species
plt.figure(figsize=(10, 6))
plt.hist(setosa['sepal length (cm)'], bins=10, alpha=0.6, label='Setosa')
plt.hist(versicolor['sepal length (cm)'], bins=10, alpha=0.6, label='Versicolor')
plt.hist(virginica['sepal length (cm)'], bins=10, alpha=0.6, label='Virginica')
plt.title('Distribution of Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

# Prepare Group Histograms
# 1. Setup the figure size and call df.hist
fig = plt.figure(figsize=(10, 6))

# df.hist returns an array of axes objects.
# We use fig.gca() to ensure it plots on the existing figure.
ax = df.hist(
    edgecolor='black',
    linewidth=1,
    ax=fig.gca(),
    bins=15,          # Specify the number of bins for detail
    grid=False        # Disable default pandas grid to customize later
)

# 2. Add an overall title
fig.suptitle('Distribution of Iris Dataset Features', fontsize=16, y=1.02)

# 3. Iterate through all subplots to apply styling
# We flatten the array of axes for easy iteration (it's a 2x2 grid)
for subplot in ax.flatten():
    # Set y-axis grid lines for readability
    subplot.grid(axis='y', linestyle='--', alpha=0.6)

    # Customizing Axes Labels
    feature_name = subplot.get_title()
    # Clean up the title/label for a professional look
    subplot.set_xlabel(f"{feature_name.replace(' (cm)', '')} (cm)", fontsize=10)
    subplot.set_ylabel("Frequency", fontsize=10)

    # Customize histogram color (using a light blue)
    for patch in subplot.patches:
        patch.set_facecolor('skyblue')


# 4. Adjust layout for perfect fit
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to account for suptitle

plt.show()
plt.savefig('beautified_df_hist.png')

# Create subplots with one row and three columns for Sepal Length
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

# Plot histograms on each subplot
'''
The alpha parameter adjusts the transparency of the elements.
It takes values between 0 (completely transparent) and 1 (completely opaque).
'''
axes[0].hist(setosa["sepal length (cm)"], bins=20, color='blue', alpha=0.7)
axes[0].set_title('Iris Setosa - Sepal Length')
axes[0].set_ylabel('Frequency')

axes[1].hist(versicolor["sepal length (cm)"], bins=20, color='green', alpha=0.7)
axes[1].set_title('Iris Versicolor - Sepal Length')

axes[2].hist(virginica["sepal length (cm)"], bins=20, color='orange', alpha=0.7)
axes[2].set_title('Iris Virginica - Sepal Length')

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plots
plt.show()

#Species-wise Histogram Visualization
# Define the features to plot
features = iris.feature_names
species = df['species'].unique()
colors = ['blue', 'green', 'orange']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Plot histograms for each feature
for i, feature in enumerate(features):
    for sp, color in zip(species, colors):
        subset = df[df['species'] == sp]
        axes[i].hist(subset[feature],
                     bins=15,
                     alpha=0.6,
                     label=sp,
                     color=color,
                     edgecolor='black',
                     linewidth=0.8)
    axes[i].set_title(f'Distribution of {feature}', fontsize=11)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

# Adjust layout for better spacing
plt.suptitle('Species-wise Histograms of Iris Flower Features', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plots
plt.show()

#Density Plot
#Plot density Plot of all numerical data
df.plot(kind = "density", figsize=(10,6))

# --- Density Plot (KDE Plot) Generation ---
#Import sns library
import seaborn as sns

#Set Figure Size
plt.figure(figsize=(10, 6))

# Use Seaborn's kdeplot.
# 'hue' separates the distributions by species.
# 'fill=True' colors the area under the curve.
sns.kdeplot(data=df,
            x='petal length (cm)',
            hue='species',
            fill=True,
            alpha=0.5,
            linewidth=2)

plt.title('Density Plot (KDE) of Petal Length by Iris Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()

# --- Species-wise Histogram with KDE Curves using Seaborn & Matplotlib ---
# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Create subplots for all features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Plot histogram + KDE for each feature
for i, feature in enumerate(features):
    sns.histplot(
        data=df,
        x=feature,
        hue='species',        # Color separation by species
        kde=True,             # Overlay KDE curves
        bins=15,
        palette='viridis',    # Attractive, colorblind-friendly palette
        alpha=0.6,
        ax=axes[i],
        edgecolor='black',
        linewidth=0.8
    )
    axes[i].set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

# Adjust layout and title
plt.suptitle('Species-wise Feature Distribution in Iris Dataset (Histogram + KDE)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show plot
plt.show()

#Visualize trend of single variable using lineplot
# --- Line Plot Code ---
plt.figure(figsize=(10, 6))

species_colors = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}

for species_name, color in species_colors.items():
    # Filter data for the current species
    species_data = df[df['species'] == species_name]

    # Plot Petal Length against its original index (sample order)
    plt.plot(species_data.index, species_data['petal length (cm)'],
             label=species_name,
             color=color,
             alpha=0.7,
             linestyle='-')

plt.title('Line Plot of Petal Length Across Dataset Samples by Species')
plt.xlabel('Sample Index')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save the plot
plt.savefig('iris_line_plot.png')
plt.show()

##Line plot using all numerical data
# default plot is a line chart
fig = plt.figure(figsize=(10, 6))

ax = df.plot() # assign to variable ax

# Use ax to add titles and labels
ax.set_title("Iris Dataset")
ax.set_xlabel("Data Points")
ax.set_ylabel("Length/Width (mm)")

#Add details to the line plot
# 1. Create the Figure and Axes objects
fig, ax = plt.subplots(figsize=(10, 6))

# 2. Iterate through columns and plot each one as a line
# Pandas plot() automatically uses the DataFrame index (0 to 149) for the x-axis.
for column in df.columns[:-1]:  # Exclude the 'species' column
    ax.plot(df.index, df[column], label=column)

# 3. Add titles and labels using the axes object
ax.set_title("Iris Dataset")
ax.set_xlabel("Data Points (Sample Index)")
ax.set_ylabel("Length/Width (cm)") # Corrected unit from 'mm' to 'cm' for Iris data

# 4. Add a legend to distinguish the lines
ax.legend(loc='upper right')

# 5. Display the plot
plt.show()

# --- Scatter Plots ---
# Scatterplot of sepal length to sepal width
plt.figure(figsize=(6,5))
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], s=40, edgecolors='k', alpha=0.7)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.show()

# Scatterplot of petal length to petal width by species
#Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot for each species
ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'],
           label='Setosa', marker='o', color='blue')
ax.scatter(versicolor['petal length (cm)'], versicolor['petal width (cm)'],
           label='Versicolor', marker='s', color='green') # 's' is a square marker
ax.scatter(virginica['petal length (cm)'], virginica['petal width (cm)'],
           label='Virginica', marker='^', color='red') # '^' is a triangle marker

# Set plot title and labels
ax.set_title('Iris Petal Length vs Petal Width by Species')
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')

# Add a legend
ax.legend(loc='upper left')

# Add grid lines for readability
ax.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()

# --- Annotation ---
# Annotate interesting points - Max petal length point
plt.figure(figsize=(6,5))
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], alpha=0.7)
# Annotate the max petal_length point
idx = df['petal length (cm)'].idxmax()
plt.annotate('max petal length', 
             xy=(df.loc[idx,'petal length (cm)'], df.loc[idx,'petal width (cm)']),
             xytext=(4.5, 1.5), 
             arrowprops=dict(arrowstyle='->'))
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.show()

# --- Multi-panel figure ---
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].scatter(df['sepal length (cm)'], df['sepal width (cm)'])
ax[0].set_title('Sepal')
ax[0].set_xlabel('Length'); ax[0].set_ylabel('Width')

ax[1].scatter(df['petal length (cm)'], df['petal width (cm)'])
ax[1].set_title('Petal')
ax[1].set_xlabel('Length'); ax[1].set_ylabel('Width')

plt.tight_layout()
plt.show()

# Combine multiple comparisons using FacetGrid - seaborn library
g = sns.FacetGrid(df, col='species', height=4)
g.map_dataframe(sns.scatterplot, x='petal length (cm)', y='petal width (cm)', color='purple', alpha=0.7)
g.set_titles('{col_name}')
g.fig.suptitle('Petal Length vs Petal Width by Species', y=1.05)
plt.show()

#COmbine Histogram and Scatterplot
# Subplots of the three species: setosa, virginica, and versicolor
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))

# First row of charts - histograms
axes[0,0].hist(setosa["sepal length (cm)"], bins=20, color='blue', alpha=0.7)
axes[0,0].set_title('Iris Setosa - Sepal Length')
axes[0,0].set_ylabel('Frequency')

axes[0,1].hist(versicolor["sepal length (cm)"], bins=20, color='green', alpha=0.7)
axes[0,1].set_title('Iris Versicolor - Sepal Length')

axes[0,2].hist(virginica["sepal length (cm)"], bins=20, color='orange', alpha=0.7)
axes[0,2].set_title('Iris Virginica - Sepal Length')

# Second row of charts - scatter plots
axes[1,0].scatter(setosa["sepal length (cm)"], setosa["sepal width (cm)"])
axes[1,0].set_title('Iris Setosa')
axes[1,0].set_xlabel('Sepal Length')
axes[1,0].set_ylabel('Sepal Width')

axes[1,1].scatter(versicolor["sepal length (cm)"], versicolor["sepal width (cm)"])
axes[1,1].set_title('Iris Versicolor')
axes[1,1].set_xlabel('Sepal Length')
axes[1,1].set_ylabel('Sepal Width')

axes[1,2].scatter(virginica["sepal length (cm)"], virginica["sepal width (cm)"])
axes[1,2].set_title('Iris Virginica')
axes[1,2].set_xlabel('Sepal Length')
axes[1,2].set_ylabel('Sepal Width')

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plots
plt.show()

# --- Box Plots ---
# Boxplot of sepal length by species (using Pandas wrapper)
plt.figure(figsize=(8,5))
df.boxplot(column='sepal length (cm)', by='species', grid=False)
plt.title('Sepal length by species')
plt.suptitle('')   # remove automatic subtitle
plt.xlabel('Species')
plt.ylabel('Sepal length (cm)')
plt.show()

# Box Plot of Sepal Width by Species (using Matplotlib raw)
sepal_width_data = [
    df[df['species'] == 'setosa']['sepal width (cm)'],
    df[df['species'] == 'versicolor']['sepal width (cm)'],
    df[df['species'] == 'virginica']['sepal width (cm)']
]
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(sepal_width_data, patch_artist=True, medianprops={'color': 'black'})
ax.set_xticklabels(df['species'].unique())
ax.set_title('Box Plot of Sepal Width by Iris Species')
ax.set_xlabel('Species')
ax.set_ylabel('Sepal Width (cm)')
ax.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()

# --- Violin Plots ---
import seaborn as sns
# Generating Violin Plot (basic)
plt.figure(figsize=(8,5))
sns.violinplot(x='species', y='petal length (cm)', data=df)
plt.title('Petal length distribution by species')
plt.show()

# Generating Violin Plot with better visualization
plt.figure(figsize=(10, 6))
sns.violinplot(
    x='species',
    y='petal length (cm)',
    data=df,
    inner='quartile', 
    palette='viridis' 
)
plt.title('Violin Plot of Petal Length by Iris Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Grouped Violin Plot
# The violinplot shows density of the length and width in the species
# Denser regions of the data are fatter, and sparser thiner in a violin plot
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
sns.violinplot(x='species', y='petal length (cm)', hue='species', data=df, inner='quartile', palette='viridis')
plt.subplot(2,2,2)
sns.violinplot(x='species', y='petal width (cm)', hue='species', data=df, inner='quartile', palette='viridis')
plt.subplot(2,2,3)
sns.violinplot(x='species', y='sepal length (cm)', hue='species', data=df, inner='quartile', palette='viridis' )
plt.subplot(2,2,4)
sns.violinplot(x='species', y='sepal width (cm)', hue='species', data=df, inner='quartile', palette='viridis' )

plt.show()

# --- Bar Chart ---
# 1. Calculate the mean petal length for each species
mean_petal_length = df.groupby('species')['petal length (cm)'].mean()

# 2. Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(mean_petal_length.index, mean_petal_length.values, color=['blue', 'green', 'red'], alpha=0.7)

# 3. Customize the plot
ax.set_title('Mean Petal Length by Iris Species')
ax.set_xlabel('Species')
ax.set_ylabel('Mean Petal Length (cm)')

# Optional: Add the mean value on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{height:.2f}',
            ha='center', va='bottom')
plt.show()

#Pie Diagram
# 1. Calculate the count of each species
species_counts = df['species'].value_counts()

# 2. Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

# The pie function takes the values and labels
ax.pie(species_counts.values,
       labels=species_counts.index,
       autopct='%1.1f%%', # Display percentages
       startangle=90,
       colors=['lightcoral', 'skyblue', 'lightgreen'],
       explode=(0.05, 0, 0), # Explode the first slice (Setosa)
       wedgeprops={'edgecolor': 'black', 'linewidth': 1}
      )

# Set the title
ax.set_title('Distribution of Iris Species in the Dataset', fontsize=16)

# Ensure the pie chart is circular
ax.axis('equal')

plt.show()

#Donut Chart Diagram
# 1. Calculate the count of each species
species_counts = df['species'].value_counts()

# 2. Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Define the properties for the donut hole (width determines the thickness of the ring)
donut_properties = {'width': 0.3, 'edgecolor': 'black', 'linewidth': 1}

# The pie function creates the segments
ax.pie(species_counts.values,
       labels=species_counts.index,
       autopct='%1.1f%%',       # Display percentages
       startangle=90,
       colors=['coral', 'skyblue', 'lightgreen'],
       explode=(0.05, 0, 0),     # Explode the first slice for emphasis
       wedgeprops=donut_properties # Creates the donut hole
      )

# Set the title
ax.set_title('Donut Chart of Iris Species Distribution', fontsize=16)

# Ensure the chart is circular
ax.axis('equal')

plt.show()

# --- Heatmaps ---
# Heat Map (using raw Matplotlib)
corr = df.iloc[:, :-1].corr()   # drop species
plt.figure(figsize=(5,4))
plt.imshow(corr, cmap='viridis', interpolation='nearest')
plt.colorbar(label='corr')
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Feature Correlation Matrix (Matplotlib)')
plt.show()

# Correlation Heatmap (using Seaborn)
numeric_df = df.drop(columns=['species'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,     
    cmap='coolwarm', 
    fmt=".2f",      
    linewidths=.5,
    linecolor='black'
)
plt.title('Correlation Heatmap of Iris Dataset Features (Seaborn)')
plt.show()

#Bubble Plot
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]
df['target'] = iris.target  # Color mapping column

#Set Image Size
plt.figure(figsize=(10, 6))

# Create the scatter (bubble) plot
scatter = plt.scatter(
    x=df['sepal length (cm)'],
    y=df['sepal width (cm)'],
    s=df['petal length (cm)'] * 50,   # Bubble size proportional to petal length
    c=df['target'],                   # Color mapped by species target
    cmap='viridis',
    alpha=0.6,
    edgecolors='w',
    linewidth=0.5
)

# Get legend handles and labels
handles, _ = scatter.legend_elements(num=3)
# Add color legend for species
plt.legend(handles, iris.target_names, title="Species", loc="lower left")

# Add size legend for petal length
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=[1, 2, 4, 6])
plt.legend(handles, ['<1 cm', '2 cm', '4 cm', '>6 cm'],
           loc="upper right", title="Petal Length (cm)")

# Add titles and labels
plt.title('Bubble Plot: Sepal Dimensions vs Petal Length', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)

# Save and display the plot
plt.savefig('iris_bubble_plot.png', bbox_inches='tight', dpi=300)
plt.show()

#Pair Plot
# Set Seaborn theme for clarity and visual appeal
sns.set(style="ticks", color_codes=True)

# Create the Pairplot
pairplot = sns.pairplot(
    df,
    hue='species',              # Color by species
    diag_kind='kde',            # KDE curve on diagonal
    palette='viridis',          # Modern color scheme
    corner=False,               # Show full upper & lower triangles
    plot_kws={'alpha': 0.6, 'edgecolor': 'k', 'linewidth': 0.5}
)

# Add a main title
pairplot.fig.suptitle('Pairplot of Iris Dataset — Feature Relationships by Species',
                      fontsize=14, fontweight='bold', y=1.02)

# Display the plot
plt.show()

#3D Scatter Plot
# Define color mapping for each species
colors = {'setosa': 'skyblue', 'versicolor': 'limegreen', 'virginica': 'coral'}

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each species separately
for species, color in colors.items():
    subset = df[df['species'] == species]
    ax.scatter(subset['petal length (cm)'],
               subset['petal width (cm)'],
               subset['sepal length (cm)'],
               label=species,
               color=color,
               s=60,
               edgecolors='k',
               alpha=0.8)

# Set axis labels
ax.set_xlabel('Petal Length (cm)', labelpad=10)
ax.set_ylabel('Petal Width (cm)', labelpad=10)
ax.set_zlabel('Sepal Length (cm)', labelpad=10)
ax.set_title('3D Scatter Plot — Iris Dataset', fontsize=14, fontweight='bold')

# Add legend and grid
ax.legend(title='Species')
ax.grid(True, linestyle='--', alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()

#Swarm Plots
plt.figure(figsize=(10, 6))

# Generate the swarm plot
sns.swarmplot(
    x='species',
    y='petal length (cm)',
    data=df,
    palette='viridis',
    size=6
)

plt.title('Swarm Plot of Petal Length by Iris Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()

#Radar Plot
# --- Radar (Spider) Plot for Iris Dataset using Matplotlib ---

df['species'] = [iris.target_names[i] for i in iris.target]

# --- Data Aggregation ---
# Calculate the mean of each feature for each species
df_avg = df.groupby('species')[df.columns[:-1]].mean().reset_index()

# Extract feature names and mean values for each species
features = df.columns[:-1]
setosa_data = df_avg.loc[df_avg['species'] == 'setosa', features].values.flatten()
versicolor_data = df_avg.loc[df_avg['species'] == 'versicolor', features].values.flatten()
virginica_data = df_avg.loc[df_avg['species'] == 'virginica', features].values.flatten()

# --- Radar Plot Setup ---
N = len(features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the circle

# Close the data circle as well
setosa_data = np.concatenate((setosa_data, [setosa_data[0]]))
versicolor_data = np.concatenate((versicolor_data, [versicolor_data[0]]))
virginica_data = np.concatenate((virginica_data, [virginica_data[0]]))

# --- Plotting ---
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Plot and fill for each species
ax.plot(angles, setosa_data, linewidth=2, linestyle='solid', label='Setosa', color='blue', alpha=0.7)
ax.fill(angles, setosa_data, color='blue', alpha=0.1)

ax.plot(angles, versicolor_data, linewidth=2, linestyle='solid', label='Versicolor', color='green', alpha=0.7)
ax.fill(angles, versicolor_data, color='green', alpha=0.1)

ax.plot(angles, virginica_data, linewidth=2, linestyle='solid', label='Virginica', color='red', alpha=0.7)
ax.fill(angles, virginica_data, color='red', alpha=0.1)

# Set axis labels
feature_labels = [f.replace(' (cm)', '').replace(' ', '\n') for f in features]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_labels, fontsize=12)

# Set radial axis range and ticks
ax.set_yticks(np.arange(0, 8, 2))
ax.set_ylim(0, 8)
ax.set_yticklabels([str(i) for i in np.arange(0, 8, 2)], color="gray", size=10)

# Add title and legend
ax.set_title('Average Feature Measurements by Iris Species', size=16, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.tight_layout()
plt.show()

#Joint Plot
# Use 'kind="scatter"' for a scatter plot with marginal histograms
sns.jointplot(x='sepal length (cm)', y='petal length (cm)', data=df, hue='species', kind='scatter', palette='viridis')
plt.suptitle('Joint Plot: Sepal Length vs. Petal Length', y=1.02)
plt.show()

# Use 'kind="kde"' for a 2D density plot with marginal KDEs
sns.jointplot(x='sepal length (cm)', y='petal length (cm)', data=df, kind='kde', fill=True, cmap='rocket')
plt.suptitle('Joint KDE Plot', y=1.02)
plt.show()

#Save Figures
plt.figure(figsize=(10,5))
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], s=40, edgecolors='k', alpha=0.7)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.savefig('iris_petal_scatter.png', dpi=300, bbox_inches='tight')
plt.close()   # close when running scripts

