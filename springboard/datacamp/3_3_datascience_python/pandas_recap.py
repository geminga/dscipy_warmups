import pandas as pd
df = your_dataset_to_dataframe

df.head() # well..head 
df.tail() # well..tail 
df.info() # tells cols, datatypes

#### Some types:
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

######## Fingerwarmup: lists to zip to dict to dataframe
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)

##### assign sensible column headers to DF
# Build a list of labels: list_labels
list_labels = ['year', 'artist', 'song', 'chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels
print(df)

##### Broadcast a constant and a list to create a new dataframe
# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

##### Assigning headers to an in-read CSV when loading in the dataframe
# Read in the file: df1
df1 = pd.read_csv(data_file)

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv(data_file, header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)

##### delimiters...headers..dropping index when saving
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

##################### BASIC STATISTICS FROM DF
# ALL SUMMARIES IGNORE NULLS
# quantile() returns median. quantile(q) returns q:th quantile.
# min and max work for strings too.
df.median() # returns ALL medians!!!
df.average() # same 

# print min and max and average across all columns
(but how would I get it only on one column?)...with "loc "
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

#### quantile API
# Print the number of countries reported in 2015
print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95])) # 5th and 95th precentile for all columns 
print(df['2015'].quantile([0.05, 0.95])) # just the column 2015

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()

# mean and std dev 
# Print the mean of the January and March data
print(january.mean(), march.mean())

# Print the standard deviation of the January and March data
print(january.std(), march.std())

# column descriptives, uniques
iris['species'].describe()
iris['species'].unique()

# access a certain column for counts
df[df['origin'] == 'Asia'].count()

#### not accepted, but is my answer correct?

# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df[df['origin'] == 'US'].count()


# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)

# IQR - Inter Quartile Range
iqr = [0.25, 0.75]
your_df.quantile(iqr)

#### Filtering rows by value in a column
# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df.loc[df['origin'] == 'US']


# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)

#### boxplots IN CASE you don't want to use dataframe.boxplot 
# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()

##### dates and times with dataframes
# grab a date/timestamp column as index
df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)

# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format=time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15':'2010-12-31']

######## Index re-indexing with fill

# Reindex without fill method: ts3
ts3 = ts2.reindex(ts1.index)

# Reindex with fill method, using forward fill: ts4
ts4 = ts2.reindex(ts1.index, method="ffill")

# Combine ts1 + ts2: sum12
sum12 = ts1 + ts2

# Combine ts1 + ts3: sum13
sum13 = ts1 + ts3

# Combine ts1 + ts4: sum14
sum14 = ts1 + ts4

#### Downsampling and upsampling
# Downsampling, e.g. from daily to weekly, upsampling opposite

# Aggregating means:
daily_mean = sales.resample('D').mean() # method chaining, like in Spark
print(daily_mean.loc['2015-02-02']) # 
sales.resample('D').mean().sum() # returns largest daily sum over data 
sales.resample('W').count() # weekly counts - also works for strings.

# OTHERS:
# min, T (minute)
# H hour 
# D day 
# B business day 
# W week 
# M month
# Q quarter
# A year

# Integer intervals for samples too
sales.loc[:,'Units'].resample('2W').sum() # aggregate over 2 weeks.

# Upsampling -> e.g. from daily to hourly 
# .. ffill() - forward fill e.g. holey data. bfill() too

# Downsample example
# Downsample to 6 hour data and aggregate by mean: df1
df1 = df.loc[:,'Temperature'].resample('6h').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df.loc[:,'Temperature'].resample('D').count()

######### examples, a dataframe has "Temperature" and other columns.


# Extract temperature data for August: august
august = df.loc['2010-08','Temperature']
# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()

# Extract temperature data for February: february
february = df.loc['2010-02','Temperature']

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample('D').min()


##### dataframe column access and new dataframe generation

#### WRONG ANSWER:
# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-08-01':'2010-08-15']

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window=24).mean()

# ERROR: smoothed and unsmoothed are both pandas.core.series.Series, not 
# dataframes, so you don't access them via loc.
# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed.loc['Temperature'], 'unsmoothed':unsmoothed.loc['Temperature']})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()

#### RIGHT ANSWER 
# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-Aug-01':'2010-Aug-15']

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window=24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()


#### month selection with partial string indexing 
# Extract the August 2010 data: august
august = df['Temperature']['2010-08']

# Resample to daily data, aggregating by max: daily_highs
daily_highs = august.resample('D').max()

# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
daily_highs_smoothed = daily_highs.rolling(window=7).mean()
print(daily_highs_smoothed)

###### Advanced filtering
pd.read_csv('your_csv.csv', parse_dates=['YourDateColumn'])
pd['col'].str.upper or lower 
pd['col'].str.contains('yourstring')

## BOOLEANS:
True + True = 2
df['col'].str.contains('yourstring').sum() # this can be done! You can sum trues

## Other 
df['YourDateColumn'].dt.hour # returns new int series with hours 
df['YourDateColumn'].dt.tz_localize('US/Central') # returns new int series with hours 
df['YourDateColumn'].dt.tz_convert('US/Central') # returns new int series with hours 
population.resample('A').first().interpolate('linear') # fills smoothly with linear interpolation! Check the rest too 

#### example - Dallas flights
# solution
# Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
# Note that this only has date and a true/false left...
dallas = df['Destination Airport'].str.contains('DAL')

# Compute the total number of Dallas departures each day: daily_departures
# ...and here we SUM BOOLEANS! That is why it works although we have a df with only dates and booleans left 
# (I did it first like this, but got scared when I had only booleans left, then did too complex things.)
daily_departures = dallas.resample('D').sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()

# In this exercise, noisy measured data that has some dropped or 
# otherwise missing values has been loaded. The goal is to compare two time series, 
# and then look at summary statistics of the differences. 
# The problem is that one of the data sets is missing data at some of the times. 
# The pre-loaded data ts1 has value for all times, yet the data set ts2 does not: 
# it is missing data for the weekends. 

# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate('linear')

# Compute the absolute difference of ts1 and ts2_interp: differences 
differences = np.abs(ts1 - ts2_interp)

# Generate and print summary statistics of the differences
print(differences.describe())

#### Time zones and conversion
# Build a Boolean mask to filter out all the 'LAX' departure flights: mask
mask = df['Destination Airport'] == 'LAX'

# Use the mask to subset the data: la
la = df[mask==True]

# Combine two columns of data to create a datetime series: times_tz_none 
times_tz_none = pd.to_datetime( la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'] )

# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

