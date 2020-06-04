
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

# Extract select column from DataFrame: col
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


#### INDEXING AND RE-INDEXING OBJECTS AND LABELED DATA 
# Re-create an index as a list 
# ehrmagerd why is this not allowed?
# Create the list of new indexes: new_idx
sales.index = sales.index.str.upper()

# Print the sales DataFrame
print(sales)

# BOOHOO NOT ACCEPTED. OK THEN.
# Create the list of new indexes: new_idx
new_idx = [i.upper() for i in sales.index]

# Assign new_idx to sales.index
sales.index = new_idx

# Print the sales DataFrame
print(sales)

# NAME COLUMN SET OF DATAFRAME 
# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTHS'

# Print the sales DataFrame
print(sales)

# Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name = 'PRODUCTS'
# Print the sales dataframe again
print(sales)

# PRODUCTS  eggs  salt  spam
# MONTHS                    
# JAN         47  12.0    17
# FEB        110  50.0    31
# MAR        221  89.0    72
# APR         77  87.0    20
# MAY        132   NaN    52
# JUN        205  60.0    55

# BUILDING AN INDEX, THEN A DATAFRAME.

# Generate the list of months: months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# Assign months to sales.index
sales.index = months

# Print the modified sales DataFrame
print(sales)

### HIERARCHICAL INDEXING - COMPOSITE / SURROGATE KEY - "MULTI-INDEX"
# aaa - 
stocks.set_index(['Symbol','Date'])

# NOTE! If this type of index, can not say "print(df.index.name)" 
# Must be print(df.index.names) 

# SORT:
your_dataframe = your_dataframe.sort_index()
# ..makes a new one actually!

# Queries are then of course requiring a reference to BOTH of the indices.
stocks.loc[('CSCO','2016-10-04')]

# Just one column (Volume) out:
stocks.loc[('CSCO','2016-10-04'), 'Volume']

# calling loc with first index component only slices the data frame.
 # and retursn all rows matching
 
 # Also : slicing can be done, this returns all from CSCO through MSFT
 stocks.loc['CSCO':'MSFT'] 
 
 # Fancy indexing - see previously, return AAPL and MSFT from the said date and from there the column 'Close'
 stocks.loc[(['AAPL','MSFT'], '2016-10-05'), 'Close']
 
 # innermost index ..
 stocks.loc[('CSCO', ['2016-10-05','2016-10-03']), :]
 
 # Both indexes requires API call: all sets of data, but only from said times.
 stocks.loc[(slice(None), slice('2016-10-03','2016-10-04')), :]

# Print individual bits and range -  
 # Print sales.loc[['CA', 'TX']]
print(sales.loc[['CA', 'TX']])

# Print sales['CA':'TX']
print(sales['CA':'TX'])


# Set multi-index and sort index ..or data by index?
 # Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state', 'month'])

# Sort the MultiIndex: sales
sales = sales.sort_index() 

# Print the sales DataFrame


# ...did I take a timecapsule to past? Set index, again...
# Set the index to the column 'state': sales
sales = sales.set_index(['state'])

# Print the sales DataFrame
print(sales)

# Access the data from 'NY'
print(sales.loc['NY'])

# ...then a version with indexing multiple levels of a multi-index.
# Just NY and month 1
print(sales.loc['NY',1])

# CA and TX and month 2
sales.loc[[('CA',2),('TX',2)]]

# Everything but only month 2 - "don't slice on first level, take "2" from second level, then return everything (:) - not a very clear API.
print(sales.loc[(slice(None),2),:])

# PIVOTING DATA FRAMES!
df.pivot(index='sarakeJokaToimiiIndeksinä',
         columns='TästäkinTuleeIndeksiJaSarakeJostaTuleeSarakkeet',
         values='ööJotainTestaa')
         
         
weekday, city, visitors, signups

# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot(
         index='weekday',
         columns='city',
         values='visitors'
    )

# Print the pivoted DataFrame
print(visitors_pivot)

## THOUGHT PAUSE:
Original data:
  weekday    city  visitors  signups
0     Sun  Austin       139        7
1     Sun  Dallas       237       12
2     Mon  Austin       326        3
3     Mon  Dallas       456        5

# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(
                 index='weekday',
                 columns='city',
                 values='signups'
            )
            
city     Austin  Dallas
weekday                
Mon           3       5
Sun           7      12

# Pivot users pivoted by both signups and visitors: pivot
        pivot = users.pivot(
                 index='weekday',
                 columns='city'
            )
            
        visitors        signups       
city      Austin Dallas  Austin Dallas
weekday                               
Mon          326    456       3      5
Sun          139    237       7     12

# Stacking dataframes 
# multi-level indexes - can't pivot. "Unstack"
trialdatadataframe.unstack(level='gender') # will take level of index 

# UNSTACKING a multi-index.
df.unstack(level='jokusarake');
# Hierarchical indexes. 

# Swapping levels
stacked.swaplevel
swapped.sort_index()

# MUSTA TÄÄ ON VAIKEA YMMÄRTÄÄ, OTAN TÄHÄN PALJON ESIMERKKEJÄ
print(users)
                visitors  signups
city   weekday                   
Austin Mon           326        3
       Sun           139        7
Dallas Mon           456        5
       Sun           237       12
       
# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level='weekday')
# OK...
print(byweekday)
        visitors      signups    
weekday      Mon  Sun     Mon Sun
city                             
Austin       326  139       3   7
Dallas       456  237       5  12

# ..sit vaan stack takaisin...
# Stack byweekday by 'weekday' and print it
print(byweekday.stack(level='weekday'))

                visitors  signups
city   weekday                   
Austin Mon           326        3
       Sun           139        7
Dallas Mon           456        5
       Sun           237       12
       
# ..mikä tän use case on? Oon pihalla.

# Ja nyt sit ei tuukkaan sama enää, mut miksi?
print(users)

                visitors  signups
city   weekday                   
Austin Mon           326        3
       Sun           139        7
Dallas Mon           456        5
       Sun           237       12

In [2]: # Unstack users by 'city': bycity
bycity = users.unstack(level='city')

# Print the bycity DataFrame
print(bycity)

# Stack bycity by 'city' and print it
print(bycity.stack(level='city'))
        visitors        signups       
city      Austin Dallas  Austin Dallas
weekday                               
Mon          326    456       3      5
Sun          139    237       7     12
                visitors  signups
weekday city                     
Mon     Austin       326        3
        Dallas       456        5
Sun     Austin       139        7
        Dallas       237       12

# more manipulations:

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))

## example: 
print(users)
                visitors  signups
city   weekday                   
Austin Mon           326        3
       Sun           139        7
Dallas Mon           456        5
       Sun           237       12

print(bycity)
        visitors        signups       
city      Austin Dallas  Austin Dallas
weekday                               
Mon          326    456       3      5
Sun          139    237       7     12

print(bycity.stack(level='city'))
                visitors  signups
weekday city                     
Mon     Austin       326        3
        Dallas       456        5
Sun     Austin       139        7
        Dallas       237       12

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort.index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))
                visitors  signups
city   weekday                   
Austin Mon           326        3
Dallas Mon           456        5
Austin Sun           139        7
Dallas Sun           237       12

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))
                visitors  signups
city   weekday                   
Austin Mon           326        3
Dallas Mon           456        5
Austin Sun           139        7
Dallas Sun           237       12
                visitors  signups
city   weekday                   
Austin Mon           326        3
       Sun           139        7
Dallas Mon           456        5
       Sun           237       12

### MELTING Dataframes 
pd.melt(dataframe, id_vars=['thecolumn'])

# specifying value_vars enables you to define what goes where.
# You can explicitly specify the columns that should remain in the reshaped DataFrame with id_vars, and list which columns to convert into values with value_vars

pd.melt(your_dataframe, id_vars=['yourcolumn'], var_name='jotain', value_name='jotain muuta')

# YET ANOTHER TRY who would ever benefit from this example?
# Reset the index: visitors_by_city_weekday
visitors_by_city_weekday = visitors_by_city_weekday.reset_index() 

# Print visitors_by_city_weekday
print(visitors_by_city_weekday)

# Melt visitors_by_city_weekday: visitors
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')

# Print visitors
print(visitors)

   city weekday  Austin  Dallas
    0        Mon     326     456
    1        Sun     139     237
      weekday    city  visitors
    0     Mon  Austin       326
    1     Sun  Austin       139
    2     Mon  Dallas       456
    3     Sun  Dallas       237
    
    # What is the use of this?
    
 # Melt users: skinny
skinny = pd.melt(users, id_vars=['visitors', 'signups'])

# Print skinny
print(skinny)

   visitors  signups variable   value
0       139        7  weekday     Sun
1       237       12  weekday     Sun
2       326        3  weekday     Mon
3       456        5  weekday     Mon
4       139        7     city  Austin
5       237       12     city  Dallas
6       326        3     city  Austin
7       456        5     city  Dallas

# Define a DataFrame skinny where you melt the 'visitors' and 'signups' columns of users into a single column.
# ..this means that you put weekday and city into vars. What the fuck?
# Melt users: skinny
skinny = pd.melt(users, id_vars=['weekday', 'city'])

# Print skinny
print(skinny)

# ..produces an unusable output.

# More unusable output 

# Set the new index: users_idx
users_idx = users.set_index(['city', 'weekday'])

# Print the users_idx DataFrame
print(users_idx)
                visitors  signups
city   weekday                   
Austin Sun           139        7
Dallas Sun           237       12
Austin Mon           326        3
Dallas Mon           456        5

# Obtain the key-value pairs: kv_pairs
kv_pairs = pd.melt(users_idx, col_level=0)

# Print the key-value pairs
print(kv_pairs)

   variable  value
0  visitors    139
1  visitors    237
2  visitors    326
3  visitors    456
4   signups      7
5   signups     12
6   signups      3
7   signups      5


Pivoting does not work if duplicates
# Pivot table works! 
# print(users  )
  weekday    city  visitors  signups
0     Sun  Austin       139        7
1     Sun  Dallas       237       12
2     Mon  Austin       326        3
3     Mon  Dallas       456        5

trials.pivot_table(index='treatment',
                    columns='gender',
                    values='response')
# In pivoting by default average is calculated, can be modified.
# aggfunc='count'

trials.pivot_table(index='treatment',
                    columns='gender',
                    values='response',
                    aggfunc='count')


# Create the DataFrame with the appropriate pivot table: by_city_day
by_city_day = users.pivot_table(
    index='weekday',
    columns='city'
    )

# Print by_city_day
print(by_city_day)

  weekday    city  visitors  signups
0     Sun  Austin       139        7
1     Sun  Dallas       237       12
2     Mon  Austin       326        3
3     Mon  Dallas       456        5

# AGGREGATION FUNCS 

# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index='weekday', aggfunc='count')

# Print count_by_weekday
print(count_by_weekday1)

# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2
count_by_weekday2 = users.pivot_table(index='weekday', aggfunc=len)
# ..note that it is not 'len'

# Verify that the same result is obtained
print('==========================================')
print(count_by_weekday1.equals(count_by_weekday2))

# Margins gives row and column sums, yeah!
# THIS ONE IS ACTUALLY USEFUL, SO, CHECK THIS:

print(users)

  weekday    city  visitors  signups
0     Sun  Austin       139        7
1     Sun  Dallas       237       12
2     Mon  Austin       326        3
3     Mon  Dallas       456        5

# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index='weekday', aggfunc=sum)

# Print signups_and_visitors
print(signups_and_visitors)

         signups  visitors
weekday                   
Mon            8       782
Sun           19       376

# Add in the margins: signups_and_visitors_total 
signups_and_visitors_total = users.pivot_table(index='weekday', aggfunc=sum, margins=True)

# Print signups_and_visitors_total
print(signups_and_visitors_total)

        signups  visitors
weekday                   
Mon            8       782
Sun           19       376
All           27      1158


# Groupby - aggregation/reduction ..counts
sales.groupby('weekday').count()
# ..creates a NEW dataframe. ...sum()
sales.groupby('weekday')['bread'].sum()

sales.groupby('weekday')[['bread','butter']].sum()

# groupby and mean: multi-level index
# ...paljonko kukin osti
sales.groupby(customer)['bread'].sum()

# Convert to category - looks bloody useless, but speeds up operations like groupby() - categories become index? 1
sales['weekday'].unique()
sales['weekday'].unique() = sales['weekday'].astype('category')

# GROUP BY SYNTAX EXAMPLES, FELT STRANGE, WHICH IS STRANGE.

# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
count_by_class = by_class['survived'].count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked','pclass'])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()

# Print count_mult
print(count_mult)

# EVEN BETTER EXAMPLE OF AGGREGATE FUNCS
# Read life_fname into a DataFrame: life and set index to country
life = pd.read_csv(life_fname, index_col='Country')

# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col='Country')

# Group life by regions['region']: life_by_region ..err group by column of another object without a join? WTF? Where did they merge?
life_by_region = life.groupby(regions['region'])

# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())

# This looks so weird, I see it as cube with splits...
# is it, column, then 2 values from there in the below example?
sales.groupby('city')[['bread','butter']].max()
# OK, ^^ groups by city and shows max values for bread and butter 

# USEFUL: Several aggregates can be passed as params see below sum, mean, count 
sales.groupby('city')[['bread','butter']].agg(['max','sum'])

# CUSTOM AGGREGATION:
# ACCEPTS ALSO UDFS AND FUNCS!!!
"aggregate by function" !!! THIS! 
see this:
def data_range(series):
    return series.max() - series.min()
    
sales.groupby('weekday')[['bread','butter']].agg(data_range)
# ..returns Monday bread had the largest range (between min & max)
        bread butter
weekday 
Mon     130   28
Sun      98   25

#...and then you can use the standard stuff in agg as well as your custom funcs..together
Here, sales dataframe, group by column customers, values bread and butter 
sales.groupby(customers)[['bread','butter']].agg({'bread':'sum', 'butter':data_range})

# MAX AND MEDIAN...
# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max','median'])

# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])

# Print the median fare in each class
print(aggregated.loc[:, ('fare','median')])


# FURTHAR COMPLETE EXAMPLES 

# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder.csv',index_col=['Year','region','Country']).sort_index()

# Group gapminder by 'Year' and 'region': by_year_region WHAT THE HELL IS THIS "LEVEL" -PARAMETER USAGE HERE?
by_year_region = gapminder.groupby(level=['Year','region'])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated 
print(aggregated.tail(6))


# YET ANOTHER - YOU GOT MIXED UP, COMPARE
# Read file: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Create a groupby object: by_day
by_day = sales.groupby(sales.index.strftime('%a'))

# Create sum: units_sum
units_sum = by_day['Units'].sum()
# You tried this (wrong)
# units_sum = by_day.groupby('Units').sum()

# Print units_sum
print(units_sum)

# TRANSFORMATIONS 
# (sd from mean - this one is good - the standardization of data, will need this)
def zscore(series):
    return (series - series.mean()) / series.std()

# DATA 
Out[3]:
mpg cyl displ hp weight accel yr origin name
0 18.0 8 307.0 130 3504 12.0 70 US chevrolet chevelle malibu
1 15.0 8 350.0 165 3693 11.5 70 US buick skylark 320
2 18.0 8 318.0 150 3436 11.0 70 US plymouth satellite
3 16.0 8 304.0 150 3433 12.0 70 US amc rebel sst
4 17.0 8 302.0 140 3449 10.5 70 US ford torino
# ...only 15 miles per gallon with biuck skylark 

# OK, then they transform with the z-score and the difference vanishes - but why is this transformation beneficial? "normalized by year"
# What?
auto.groupby('yr')['mpg'].transform(zscore).head()
    mpg 
0 0.058125
1 -0.503753
2 0.058125
3 -0.316460
4 -0.129168

# Then the same to a def
def zscore_with_year_and_name(group):
    df = pd.DataFrame(
        {'mpg': zscore(group['mpg']),
        'year': group['yr'],
        'name': group['name']}
        )
return df

# ..a def apparently is not eaten by transform, so you need "apply".
auto.groupby('yr').apply(zscore_with_year_and_name).head()

mpg name year
0 0.058125 chevrolet chevelle malibu 70
1 -0.503753 buick skylark 320 70
2 0.058125 plymouth satellite 70
3 -0.316460 amc rebel sst 70
4 -0.129168 ford torino 70

# USEFUL: filter data by outliers
# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby('region')['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers (VERY low life, VERY high fertility)
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)

# Inputting missing data 
# Dealing with missing data is natural in pandas (both in using the default behavior and in defining a custom behavior). 

# Filling missing data (imputation) by group

# Many statistical and machine learning packages cannot determine the best action to take when missing data entries are encountered. Dealing with missing data is natural in pandas (both in using the default behavior and in defining a custom behavior). In Chapter 1, you practiced using the .dropna() method to drop missing values. Now, you will practice imputing missing values. You can use .groupby() and .transform() to fill missing data appropriately for each group.

# Your job is to fill in missing 'age' values for passengers on the Titanic with the median age from their 'gender' and 'pclass'. To do this, you'll group by the 'sex' and 'pclass' columns and transform each group with a custom function to call .fillna() and impute the median value.

# The DataFrame has been pre-loaded as titanic. Explore it in the IPython Shell by printing the output of titanic.tail(10). Notice in particular the NaNs in the 'age' column.


    # Group titanic by 'sex' and 'pclass'. Save the result as by_sex_class.
    # Write a function called impute_median() that fills missing values with the median of a series. This has been done for you.
    # Call .transform() with impute_median on the 'age' column of by_sex_class.
    # Print the output of titanic.tail(10). This has been done for you - hit 'Submit Answer' to see how the missing values have now been imputed.

# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age'] ...butbutbut it can't be, impute_median expects arg..so the series flows implicitly from the by_sex_class one to an argument? WHAAAT?
titanic.age = by_sex_class['age'].transform(impute_median)
# OR 
# titanic.age = by_sex_class.age.transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))


# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

# Apply the disparity function on regional: reg_disp - said function has parameter (rg)..but here you don't give it any parameter. WTF?
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])
# FOR FUCKS SAKES WHY IS THIS NOT GOOD YOU FUCKING API FUCK: print(reg_disp.loc['United States','United Kingdom','China'])

# I hate this bracket-dance, groupby has ( and then if applying func then you have [] for the grouped and func.
# OK, this is for yearly average grouped by year, then you count mean of mpg.
my_auto_dataframe.groupby('year')['mpg'].mean()

# ..but what if we want that only for cars built by Chevrolet.
# ...filter first, then group by. OK.

split_df = my_auto_dataframe.groupby('year')['mpg'].mean()

# why a "group by" *object*? Why not a new df with the grouped data?
# ANOTHER EXAMPLE
# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)


### A GOOD TEST USE CASE
# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
print(by_com_filt)


# Ei vaan vittu jaksa.

# Filtering and grouping with .map()

# You have seen how to group by a column, or by multiple columns. Sometimes, you may instead want to group by a function/transformation of a column. The key here is that the Series is indexed the same way as the DataFrame. You can also mix and match column grouping with Series grouping.

# In this exercise your job is to investigate survival rates of passengers on the Titanic by 'age' and 'pclass'. In particular, the goal is to find out what fraction of children under 10 survived in each 'pclass'. You'll do this by first creating a boolean array where True is passengers under 10 years old and False is passengers over 10. You'll use .map() to change these values to strings.

# Finally, you'll group by the under 10 series and the 'pclass' column and aggregate the 'survived' column. The 'survived' column has the value 1 if the passenger survived and 0 otherwise. The mean of the 'survived' column is the fraction of passengers who lived.

# The DataFrame has been pre-loaded for you as titanic.

# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10, 'pclass'])['survived'].mean()
print(survived_mean_2)


# # # The olympic medal dataset with index 
USA_edition_grouped = medals.loc[medals.NOC == 'USA'].groupby('Edition')

# Select the 'NOC' column of medals: country_names (but how do I get countries in then?)
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts (why is it not count() here but value_counts()?)
medal_counts = country_names.value_counts() 

# Print top 15 countries ranked by medals
print(medal_counts.head(15))


#### Pivot table, aggregating by count 
# Construct the pivot table: counted
counted = medals.pivot_table(index='NOC',
                    columns='Medal',
                    values='Athlete',
                    aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))

### REMOVE DUPLICATES, PRINT UNIQUES
# Select columns: ev_gen
ev_gen = medals[['Event_gender', 'Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)

# # Find errors with .groupby()
# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender','Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)

                     City  Edition  Sport  Discipline  Athlete    NOC  Event  Medal
Event_gender Gender                                                                 
M            Men     20067    20067  20067       20067    20067  20067  20067  20067
W            Men         1        1      1           1        1      1      1      1
             Women    7277     7277   7277        7277     7277   7277   7277   7277
X            Men      1653     1653   1653        1653     1653   1653   1653   1653
             Women     218      218    218         218      218    218    218    218

### Find out offending row 
# Create the Boolean Series: sus
sus = (medals['Event_gender'] == 'W') & (medals['Gender'] == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals.loc[sus]

# Print suspect
print(suspect)

        City  Edition      Sport Discipline            Athlete  NOC Gender     Event Event_gender   Medal
23675  Sydney     2000  Athletics  Athletics  CHEPCHUMBA, Joyce  KEN    Men  marathon            W  Bronze

# Ranking of distinct events, top file countries winning medals and comparing cold war rivals
idxmax() - row or column label where max
idxmin() - row or column label where min

weather.idxmax() - if table with MONTH column and months in rows, will return monthname with max temp.

If months in columns and values underneath them, put idxmax(axis='columns')
weather.T.idxmax (T maybe temp?)


# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))

### Select data with two boolean series THIS LOOKS USEFUL 
# Create a Boolean Series that is True when 'Edition' is between 1952 and 1988: during_cold_war
during_cold_war = (medals['Edition'] >= 1952) & (medals['Edition'] <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA','URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)

# Print Nsports
print(Nsports)


### Check out how many times a country in USA, URS got most medals during cold war 

# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index='Edition',
columns='NOC', values='Athlete', aggfunc='count')

# Slice medals_won_by_country: cold_war_usa_urs_medals
cold_war_usa_urs_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]

# Create most_medals 
most_medals = cold_war_usa_urs_medals.idxmax(axis='columns')

# Print most_medals.value_counts()
print(most_medals.value_counts())

### PLOTS FROM DATAFRAMES 
Remember that *indexes are handy, since they end up as plot axis labels*

Problem: Groupby creates multi-level indices, and matplotlib does not handle them well...so you will need the bloody "unstack"
"reshaping"
Distinct columns create distinct lineplots, hence

IMPORTANT:
Use indexes, they will be the axes.
Put your data to columns by the index, you will get separate graphs easily.
(why use Python as a plotting tool?)

EN OIS OSANNU
# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()

...creates a line plot.

Now an area plot.


# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()

# # # CATEGORIES IN THE ORDER YOU WANT TO THE GRAPH 
# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values = medals.Medal, categories=['Bronze', 'Silver', 'Gold'], ordered=True)

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()

