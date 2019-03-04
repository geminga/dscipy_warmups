# GENERALLY - I wonder why anyone would use python visualizations, if one can avoid them?
# PowerBI destroys all this. In an iPython notebook, yeah, sure.

#################################
# PLOTTING EXAMPLES
#################################
# GENERAL
# show() shows the diagram

#################################
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()

#################################
# Multiple axes - creates two plots side by side - tick labels totally borken though
# Create plot axes for the first line plot
plt.axes([0.05, 0.05, 0.425, 0.9])

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Create plot axes for the second line plot
plt.axes([0.525, 0.05, 0.425, 0.9])

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()


#################################
# "subplot" - vastly superior looking - 2x2 etc - this way plot clusters will end up looking superior 


# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1,2,1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1,2,2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()

##################################
# HERE A 2X2 SUBPLOT MATRIX - REMEMBER FROM TOP LEFT ROW-WISE
# Create a figure with 2x2 subplot layout and make the top left subplot active
plt.subplot(2,2,1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the top right subplot active in the current 2x2 subplot grid 
plt.subplot(2,2,2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Make the bottom left subplot active in the current 2x2 subplot grid
plt.subplot(2,2,3)

# Plot in green the % of degrees awarded to women in Health Professions
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Make the bottom right subplot active in the current 2x2 subplot grid
plt.subplot(2,2,4)

# Plot in yellow the % of degrees awarded to women in Education
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()


##################################
# X AXIS SCALING AND "ZOOM" - also 2 graphs in same plot..

# Zooming to axis: one command gives the xlow, xhigh, ylow, yhigh
plt.axis((1947,1957,0,600))
# Check out axis commands, e.g. axis('equal'), 

axis('off')
axis('equal') # ensures that equal length
axis('square')
axis('tight') # sets xlim(), ylim() to show all data

# Plot the % of degrees awarded to women in Computer Science and the Physical Sciences
plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')

# Add the axis labels
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')

# Set the x-axis range
plt.xlim(1990, 2010)

# Set the y-axis range
plt.ylim(0, 50)

# Add a title and display the plot
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()

# Save the image as 'xlim_and_ylim.png'
plt.savefig('xlim_and_ylim.png')


#### More concise:
# Plot in blue the % of degrees awarded to women in Computer Science
plt.plot(year,computer_science, color='blue')

# Plot in red the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color='red')

# Set the x-axis and y-axis limits
plt.axis((1990,2010,0,50))

# Show the figure
plt.show()

# Save the figure as 'axis_limits.png'
plt.savefig('axis_limits.png')


##################################
# legend() ! 

# ,,,,

# Specify the label 'Computer Science'
plt.plot(year, computer_science, color='red', label='Computer Science') 

# Specify the label 'Physical Sciences' 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences') 


# Add a legend at the lower center
plt.legend(loc='lower center')

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()

# ...it plotted the labels several times...this is borken

################ ANNOTATE WITH ARROW ON HIGHEST POINT: 
# Plot with legend as before
plt.plot(year, computer_science, color='red', label='Computer Science') 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()

# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.argmax()]

# Add a black arrow annotation
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max+5, cs_max+5), arrowprops=dict(facecolor='black'))

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()


################# this one does a plot matrix
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Set the style to 'ggplot'
plt.style.use('ggplot')

# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1) 

# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()


###### RASTER PLOT MESH
Basic idea: visualizing numpy array of color values.
meshgrid()

# Import numpy and matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

# Generate two 1-D arrays: u, v
u = np.linspace(-2, 2, 41)
v = np.linspace(-1, 1, 21)

# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u, v)

# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 

# Display the resulting image with pcolor()
plt.pcolor(Z)
plt.show()

# Save the figure to 'sine_mesh.png'
plt.savefig('sine_mesh.png')


######### More 2 D array stuff
## remember - it starts left to right bottom up
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 2, 1], [0, 0, 1], [-1, 1, 1]])
plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()

A = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])
plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()

A = np.array([[-1, 0, 1], [1, 0, 2], [1, 1, 1]])
plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()

A = np.array([[1, 1, 1], [2, 0, 1], [1, 0, -1]])
plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()

plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()

# p-colour = pseudocolor (?) - 
import numpy as np 
import matplotlib.pyplot as plt 
u = np.linspace(-2, 2, 65)
v = np.linspace(-1, 1, 33)
X,Y = np.meshgrid(u, v)
Z = X**2/25 + Y**2/4
plt.pcolor(Z)
plt.pcolor(Z, cmap= 'autumn') 
plt.colorbar()
# plt.axis('tight') - kato mitä tää tekee
plt.show()

# check out other cmaps - documentatoin here
https://matplotlib.org/examples/color/colormaps_reference.html

# contour plots better for continuous vars 

plt.pcolor(X, Y, Z) # X, Y are 2D mesh grid
plt.colorbar()
plt.show()

plt.contour(Z)
plt.show()

# n of contours 
plt.contour(Z, 30)
plt.show()

# FILLED filled contour - it just has "f" there in addition
plt.contourf(X, Y, Z, 30) 
plt.colorbar()
plt.show()

# MOAR: 
matplotlib.pyplot documentation
http://matplotlib.org/gallery.html

# EXAMPLES:
# Generate a default contour map of the array Z (x,y - asettaa origon keskelle)
plt.subplot(2,2,1)
plt.contour(X, Y, Z)

# Generate a contour map with 20 contours
plt.subplot(2,2,2)
plt.contour(X, Y, Z, 20)

# Generate a default filled contour map of the array Z
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)

# Generate a default filled contour map with 20 contours
plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 20)

# Improve the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()


###############
# Create a filled contour plot with a color map of 'viridis'
plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')

# Create a filled contour plot with a color map of 'gray'
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')

# Create a filled contour plot with a color map of 'autumn'
plt.subplot(2,2,3)
plt.contourf(X,Y,Z,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')

# Create a filled contour plot with a color map of 'winter'
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20, cmap='winter')
plt.colorbar()
plt.title('Winter')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()

###### STATISTICAL GRAPHS
# histograms - bins
plt.hist2d(x, y, bins=(10, 20))

# about the syntax below: 
#           x  y         x , y         xmin, xmax, ymin, ymax
plt.hist2d(hp, mpg, bins=(20,20), range=((40, 235), (8, 48)))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()

### SAME AS HEXPLOT
# why hex? 
# https://cran.r-project.org/web/packages/hexbin/vignettes/hexagon_binning.pdf
# Generate a 2d histogram with hexagonal bins
plt.hexbin(hp, mpg, gridsize=(15,12), extent=((40, 235, 8, 48)))
           
# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()

# scatterplot 
# 2-D histogram is very cool! Shows concentration of 
# data with colour as an additional dimension

########## SEABORN
# works well with Pandas dataframes

# basics - imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#### A regression: 
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x='weight', y='hp', data=auto)

# Display the plot
plt.show()

#### Residual plot 
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')

# Display the plot
plt.show()

#### Scatter plot - check out the "order" for the regression lines
# dataset is called "auto", so don't get confused
# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')

# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, order=1, color='blue', scatter=None, label='order 1')

# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, order=2, color='green', scatter=None, label='order 2')

# Add a legend and display the plot
plt.legend(loc='upper right')
plt.show()

####### Grouping by hue
# Often it is useful to compare and contrast trends between different groups. Seaborn makes it possible to apply linear regressions separately for subsets of the data by applying a groupby operation. Using the hue argument, you can specify a categorical variable by which to group data observations. The distinct groups of points are used to produce distinct regressions with different hues in the plot.

# Very powerful: regression grouped by variable - with scatterplot, with regressionlines

# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x='weight', y='hp', data=auto, hue='origin',palette='Set1')

# Display the plot
plt.show()

## grouping regressions!!!!!!!!!!!!!!!!!!
# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x='weight', y='hp', data=auto, hue='origin',palette='Set1', row='origin')

# Display the plot
plt.show()

## univariate plotting
# Strip plot shows outliers well: 
sns.stripplot(y= 'your_var', data=yourdataframe)
plt.ylabel(' your label here')
plt.show()

# grouped stripplot:
sns.stripplot(x='your_grouping_x_column_var', y='your_var_to_be_grouped', data=your_dataset)
...horizontal jitter helps.

# Swarm plot is even better 
sns.swarmplot(x='your_grouping_x_column_var', y='your_var_to_be_grouped', data=your_dataset)
plt.ylabel(' your label here')
plt.show()

# Swarm plot here with a hue sub-grouping and data as horizontal NOTE THE AXES!
ns.swarmplot(x='your_var_to_be_grouped', y='your_grouping_column_var', data=your_dataset, hue='your_split', orient='h')
plt.ylabel(' your label here')
plt.show()

# when lotsa data, (box plots) and violins
# Violins use Kernel density estimates instead of blocky histograms!
plt.subplot(1,2,1)
sns.boxplot(x='your_grouping_column_var', y='your_var_to_be_grouped', data=your_dataset)
plt.ylabel(' your label here')
plt.subplot(1,2,2)
sns.violinplot(x='your_grouping_column_var', y='your_var_to_be_grouped', data=your_dataset)
plt.ylabel(' your label here')
plt.tight_layout()
plt.show()

# swarmplot overlayed with violin


################## IDEAS!! POSTAL CODE PROFILE
# 1) heatmap of correlations, so that it actually works 
# 2) regression plots of correlating variables...grouped by...year?
# 3) ...with residual plot 
# 4) seaborn's hex-histogram

sns.violinplot(x='your_grouping_column_var', y='your_var_to_be_grouped', data=your_dataset, inner=None, color='lightblue')
sns.stripplot(x='your_grouping_x_column_var', y='your_var_to_be_grouped', data=your_dataset, size=4, jitter=True)
plt.ylabel(' your label here')
plt.show()
