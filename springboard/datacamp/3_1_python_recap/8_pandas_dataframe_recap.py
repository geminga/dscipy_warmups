
# INDEX, SERIES, DATAFRAME 
# INDEX: series of labels.
# SERIES: 1D array with labels.
# DataFrame: 2D array with Series as columns 

# INDEX:
# You can slice indices like any series 
# renaming index
# Index = immutable (of course you can overwrite the index by re-assigning a new index on it.
yourdataframe.index.name = 'your_new_name'
yourdataframe.index = yourdataframe['the_col_you_woant']
# get index right the first time 
my_df = pd.read_csv('my_csv_to_dataframe',index_col = 'whateveryouwant')

# Reading and using Pandas dataframes
# The dataset contains Twitter data and you will iterate over entries in a column to build a dictionary in which the keys are the names of languages and the values are the number of tweets in the given language. The file tweets.csv is available in your current directory.

# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)

# Other things I did
# Dataframe contents
df

# Got suspicious of dataframe structure, so checked schema
# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)
# .. schema looked OK and believable

# Got suspicious of col contents, so printed it 
print(col)
# ..strange content, what language is "et"? und? is that "undefined"? Crappy stuff.

# Answer was correct anyhoo, so OK.

#### ACCESSES..ACCESSES..BY INDEX, BY NAME...
## THIS:
# read csv, set index: 
df = pd.read_csv('sales.csv', index_col='month')

month|eggs|salt|spam
Jan
Feb
Mar 
# you get it ^

# So...
df['salt']['Jan'] # NOTE: in here first x, then y. Why, don't know.
df.eggs['Mar']
df.loc['May', 'spam'] 
df.iloc[4,2] # row index, column index, starts from 0
new_sub_df = df[['salt','eggs']]

# EXAMPLE:
# Assign the row position of election.loc['Bedford']: x
x = 4

# Assign the column position of election['winner']: y
y = 4

# Print the boolean equivalence
print(election.iloc[x, y] == election.loc['Bedford', 'winner'])


# Import pandas
import pandas as pd

# Read in filename and set the index: election
election = pd.read_csv(filename, index_col='county')

# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results
results = election[['winner', 'total', 'voters']]

# Print the output of results.head()
print(results.head())

#### FILTERING - 
## Create a series of booleans 
# Just listing 
df[df.salt_amount > 60] # just prints out those for which is true

# ..assigning to a df-var for later usage: 
enough_salt = df.salt > 60
df[enough_salt]

# remember that you have and and or offered too
df[(df.salt >= 50) & (df.eggs < 200)] # AND here 
df[(df.salt >= 50) | (df.eggs < 200)] # OR here 

# selecting non-zeros
df.all() # returns only data, which is complete along investigated axis - returns Series, unless a level is specified. Only complete data is returned.

df.any() # returns True if there are any non-null, non-zero values or trues 
# 

df.any() # having any non-zero value 

# to find cols with nulls:
df.loc[:, df.isnull().any()] # ..all columns, any for which isnull = true.

# to get only complete data:
df.loc[:, df.notnull().all()] # since all() looks at boolean Trues, this works.

# to drop rows with any NaN:s 
df.dropna(how='any') #

# Modifying a column based on another
df.eggs[df.salt > 55] += 5 # for each eggs - col, add 5 if salt > 55

### Examples:
# In this exercise, we have provided the Pennsylvania election results and included a column called 'turnout' that contains the percentage of voter turnout per county. Your job is to prepare a boolean array to select all of the rows and columns where voter turnout exceeded 70%.

# As before, the DataFrame is available to you as election with the index set to 'county'.
# Create the boolean array: high_turnout
high_turnout = election.turnout > 70

# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election[high_turnout]

# Print the high_turnout_results DataFrame
print(high_turnout_df)

#### assign np.nan to "winner" where it was actually too close to call 
# Import numpy
import numpy as np

# Create the boolean array: too_close
too_close = election.margin < 1 

# Assign np.nan to the 'winner' column where the results were too close to call
election.winner[too_close] = np.nan

# Print the output of election.info()
print(election.info())

#### Drop columns which have less than 1000 non-missing values 
# Select the 'age' and 'cabin' columns: df
df = titanic[['age','cabin']]

# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
print(df.dropna(how='any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how='all').shape)

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())

#### DataFrame transformations - without loops!
## Pandas native, if not works, then
## numpy u-funcs in addition of dataframe's methods 
df.floordiv(12) # convert the whole dataframe to  dozens unit! 
np.floordiv(df, 12) # convert to dozens unit with NumPy

# you could also make your own def - then call it with df.apply(your_func) 
# Example: 
def dozens(n):
    return n//12 # (why 2 times / ? )

# then, 
df.apply(dozens)

# ..and then of course a lambda, way more compact way to use "throwaway functions"
df.apply(lambda n: n//12)

## transformation stored to columns based on another column, here column = 'eggs'
df['dozens_of_eggs'] = df.eggs.floordiv(12)
df['dozens_of_eggs'] = df.eggs.apply(lambda n: n//12)

#### String values transformations
df.index # gives you innards of index. 
# .str is a handy accessor for vectorized string transformations 
# Here e.g. a string index to uppercase:
df.index = df.index.str.upper()

#for INDEX there IS NOT APPLY method. Sadly, there is "map".
df.index = df.index.map(str.lower) # I don't get this - the example above handles index diractly, what does it matter that I don't have "apply " available? What is wrong with using the .str? Don't get the relevance here.

# Defining columns as result of other columns 
df['salty_eggs'] = df.salt + df.eggs

# DON'T LOOP IF YOU CAN USE PANDAS VECTORIZED COMPS FROM PANDAS API

# EXAMPLE: APPLY TO DF OVER SELECTED COLUMNS (OR EVERYTHING?)
# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF','Mean Dew PointF']].apply(to_celsius)

# Rename the columns df_celsius
df_celsius = df_celsius.rename(columns={'Mean TemperatureF':'Mean TemperatureC', 'Mean Dew PointF':'Mean Dew PointC'})

# Print the output of df_celsius.head()
print(df_celsius.head())

#### USING MAP 
# Schema: 
# Index(['state', 'total', 'Obama', 'Romney', 'winner', 'voters'], dtype='object')

# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue','Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election.winner.map(red_vs_blue)

# Print the output of election.head()
print(election.head())

### ON PERFORMANCE
# When performance is paramount, you should avoid using .apply() and .map() because those constructs perform Python for-loops over the data stored in a pandas Series or DataFrame. 
# ***By using vectorized functions instead, you can loop over the data at the same speed as compiled code (C, Fortran, etc.)! ***

# FIND LIST OF VECTORIZED FUNCS NumPy, SciPy and pandas come with a variety of vectorized functions (called Universal Functions or UFuncs in NumPy).

## Z-score usage 
# Import zscore from scipy.stats
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())


#### INDEX OBJECTS AND LABELED DATA 
