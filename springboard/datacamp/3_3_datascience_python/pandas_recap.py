import pandas as pd
df = your_dataset_to_dataframe

df.head() # well..head 
df.tail() # well..tail 
df.info() # tells cols, datatypes

## Some types:
# Import numpy
import numpy as np

# Create array of DataFrame values: np_vals
np_vals = df.values

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]

np_vals = numpy.ndarray
np_vals_log10 = numpy.ndarray
df = pandas.core.frame.DataFrame

# read: 
pd.read_csv

# Dataframes from dict - first column -> index 
# How to get data into dataframes (e.g. from dict)
# dataframes from lists too
So.

# Dataframe from list of lists 
import pandas as pd
cities = ['Helsinki', 'Turku', 'Tampere', 'Lahti']
signups = [7, 12, 3, 5]
visitors = [139, 237, 326, 456]
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
list_labels = ['city',  'signups', 'visitors', 'weekday']
list_cols = [cities,  signups, visitors, weekdays]
zipped = list(zip(list_labels, list_cols))

print(zipped)

# from zip to dict 
data = dict(zipped)
users = pd.DataFrame(data)
print(users)

# Broadcasting: numpy & pandas
users['fees'] = 0 # broadcasts to entire column -> now users has a new column with val 0

# extending an existing dataframe
import pandas as pd
heights = [85.0, 84.5, 94.0, 30.3]
data = {'height': heights, 'sex': 'M'} # this glues the heights into existing dataframe AND creates a new column 'sex' with all entries as 'M'
results = pd.DataFrame(data)

# Both index and columns can be altered
results.columns = ['height (inches)', 'gender']
results.index = ['A','B','C','D']

#### Fingerwarmup: lists to zip to dict to dataframe
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)

### assign sensible column headers to DF
# Build a list of labels: list_labels
list_labels = ['year', 'artist', 'song', 'chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels
print(df)

### Broadcast a constant and a list to create a new dataframe
# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

### Assigning headers to an in-read CSV when loading in the dataframe
# Read in the file: df1
df1 = pd.read_csv(data_file)

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv(data_file, header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)

### delimiters...headers..dropping index when saving
# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv('file_clean', index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('file_clean.xlsx', index=False)

########### BASIC STATISTICS FROM DF
# ALL SUMMARIES IGNORE NULLS
# quantile() returns median. quantile(q) returns q:th quantile.
# min and max work for strings too.
df.median() # returns ALL medians!!!
df.average() # same 

# print min and max and average across all columns
(but how would I get it only on one column?)
# Print the minimum value of the Engineering column
print(min(df['Engineering']))

# Print the maximum value of the Engineering column
print(max(df['Engineering']))

# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')


# Plot the average percentage per year
mean.plot()

# Display the plot
plt.show()


# Print summary statistics of the fare column with .describe()
print(df['fare'].describe())

# Generate a box plot of the fare column
df['fare'].plot(kind='box')

# Show the plot
plt.show()
