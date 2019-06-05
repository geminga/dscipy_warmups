# datetime module, jonka sisällä datetime
.striptime()

from datetime import datetime
print(parking_violations_date)

date_dt = datetime.strptime(parking_violations_date, '%m/%d/%Y')
print(date_dt)

# from datetime object to string
date_dt.strftime('%m/%d/%Y')
date_dt.isoformat()

# Import the datetime object from datetime
from datetime import datetime

# Iterate over the dates_list 
for date_str in dates_list:
# Convert each date to a datetime object: date_dt
date_dt = datetime.strptime(date_str, '%m/%d/%Y')

# Print each date_dt
print(date_dt.isoformat())

# Loop over the first 10 items of the datetimes_list
for item in datetimes_list[:10]:
# Print out the record as a string in the format of 'MM/DD/YYYY'
print(item.strftime('%m/%d/%Y'))

# Print out the record as an ISO standard string
print(item.isoformat())

Datetimes great for grouping

# Remember
.now()
.utcnow()

pytz module via timezone object 
Timezone aware objects have .astimezone()

from pytz import timezone 

In [3]: ny_tz = timezone('US/Eastern')
In [4]: la_tz = timezone('US/Pacific')
In [5]: ny_dt = record_dt.replace(tzinfo=ny_tz)

# Create a defaultdict of an integer: monthly_total_rides
monthly_total_rides = defaultdict(int)

# Loop over the list daily_summaries
for daily_summary in daily_summaries:
# Convert the service_date to a datetime object
service_datetime = datetime.strptime(daily_summary[0], '%m/%d/%Y')

# Add the total rides to the current amount for the month
monthly_total_rides[service_datetime.month] += int(daily_summary[4])

# Print monthly_total_rides
print(monthly_total_rides)


# you can make a datetime object "aware" by passing a timezone as the tzinfo keyword argument to the .replace() method on a datetime instance.

# EDUCATIONAL TIMEZONE DEMO
from pytz import timezone
# Create a Timezone object for Chicago
chicago_usa_tz = timezone('US/Central')

# Create a Timezone object for New York
ny_usa_tz = timezone('US/Eastern')

# Iterate over the daily_summaries list
for orig_dt, ridership in daily_summaries:

# Make the orig_dt timezone "aware" for Chicago
chicago_dt = orig_dt.replace(tzinfo=chicago_usa_tz)

# Convert chicago_dt to the New York Timezone
ny_dt = chicago_dt.astimezone(ny_usa_tz)

# Print the chicago_dt, ny_dt, and ridership
print('Chicago: %s, NY: %s, Ridership: %s' % (chicago_dt, ny_dt, ridership))

#######################################
# Incrementing through time
#######################################
# timedelta

In [1]: from datetime import timedelta
In [2]: flashback = timedelta(days=90)
In [3]: print(record_dt)
2016-07-12 04:39:00
In [4]: print(record_dt - flashback)
2016-04-13 04:39:00
In [5]: print(record_dt + flashback)
2016-10-10 04:39:00

# between
timedelta is a type!

In [1]: time_diff = record_dt - record2_dt
In [2]: type(time_diff)
Out[2]: datetime.timedelta
In [3]: print(time_diff)
0:00:04

# Import timedelta from the datetime module
from datetime import timedelta

# Build a timedelta of 30 days: glanceback
glanceback = timedelta(days=30)

# Iterate over the review_dates as date
for date in review_dates:
# Calculate the date 30 days back: prior_period_dt
prior_period_dt = date - glanceback

# Print the review_date, day_type and total_ridership
print('Date: %s, Type: %s, Total Ridership: %s' %
(date, 
daily_summaries[date]['day_type'], 
daily_summaries[date]['total_ridership']))

# Print the prior_period_dt, day_type and total_ridership
print('Date: %s, Type: %s, Total Ridership: %s' %
(prior_period_dt, 
daily_summaries[prior_period_dt]['day_type'], 
daily_summaries[prior_period_dt]['total_ridership']))

# DateArithmetic = automatically timedelta

from datetime import timedelta
# Iterate over the date_ranges
for start_date, end_date in date_ranges:
# Print the End and Start Date
print(end_date, start_date)
# Print the difference between each end and start date
print(end_date - start_date)

type(end_date - start_date) 
datetime.timedelta

############# PENDULUMN LIBRARY - timestamps with timezone and all

.parse()

import pendulum
occurred = violation[4] + ' ' + violation[5] +'M'
occurred_dt = pendulum.parse(occurred, tz='US/Eastern')
print(occured_dt)
'2016-06-11T14:38:00-04:00'

print(pendulum.now('Asia/Tokyo'))
<Pendulum [2017-05-06T08:20:40.104160+09:00]>

print(diff.in_words())
.in_days()
.in_hours()

### Pendulum's now etc.
# Import the pendulum module
import pendulum

# Create a now datetime for Tokyo: tokyo_dt
tokyo_dt = pendulum.now(tz='Asia/Tokyo')

# Covert the tokyo_dt to Los Angeles: la_dt
la_dt = tokyo_dt.in_timezone('America/Los_Angeles')

# Print the ISO 8601 string of la_dt
print(la_dt.to_iso8601_string())

############ 30/02/2019 can be parsed just by saying "strict = false"
## Otherwise expects ISO-dates
# Iterate over date_ranges
for start_date, end_date in date_ranges:

# Convert the start_date string to a pendulum date: start_dt 
start_dt = pendulum.parse(start_date, strict = False)

# Convert the end_date string to a pendulum date: end_dt 
end_dt = pendulum.parse(end_date, strict = False)

# Print the End and Start Date
print(end_dt, start_dt)

# Calculate the difference between end_dt and start_dt: diff_period
diff_period = end_dt - start_dt

# Print the difference in days
print(diff_period.in_days())

###### calculating timedelta for daily trip allowance
import pendulum

# Convert the start_date string to a pendulum date: start_dt 
start_dt = pendulum.parse('2019-05-19T13:00', strict = False)

# Convert the end_date string to a pendulum date: end_dt 
end_dt = pendulum.parse('2019-05-26T20:00', strict = False)


# Print the End and Start Date
print(end_dt, start_dt)

# Calculate the difference between end_dt and start_dt: diff_period
diff_period = end_dt - start_dt

# Print the difference in days
print('Days: ')
print(diff_period.in_days())
print('Hours: ')
print(diff_period.in_hours())
print(diff_period.in_hours()/24)


