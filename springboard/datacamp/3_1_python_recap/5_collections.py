# "container sequences"
#################################
# Lists
#################################

# Create a list containing the names: baby_names
baby_names = ['Ximena', 'Aliza', 'Ayden', 'Calvin']

# Extend baby_names with 'Rowen' and 'Sandeep'
baby_names.extend(['Rowen', 'Sandeep'])

# Print baby_names
print(baby_names)

# Find the position of 'Aliza': position
position = baby_names.index('Aliza')

# Remove 'Aliza' from baby_names
baby_names.pop(position)


# Some more iterations

# Create the empty list: baby_names
baby_names = []

# Loop over records 
for row in records:
    # Add the name to the list
    baby_names.append(row[3])
print(baby_names)
# Sort the names in alphabetical order
for name in sorted(baby_names):
    # Print each name
    print(name)

#################################
# Tuples (in databases!)
#################################

index to 
immutable
more efficient than list
pair elements
unpackable

zipping and unpacking
[(item, item),(blaa,blaa)]

Tuples are commonly created by zipping lists together with zip(list,list,list..)

Unpacking: 
us_num_1, in_num_1 = top_pairs[0]
print(us_num_1) # tulee ton pairin eka
print(in_num_1) # tulee ton pairin toka
...eli nimellä voi viitata.

# Unpacking in loops
for us_cookie, in_cookie in top_pairs:
    print(in_cookie)
    print(us_cookie)
    
# Enumerating positions of tuples
for idx, item in enumerate(top_pairs):
    us_cookie, in_cookie = item
    print(idx, us_cookie, in_cookie)
(0, 'Chocolate Chip', 'Punjabi')
(1, 'Brownies', 'Fruit Cake Rusk')

# Note zip(), enumerate(), () create tuple

# Using and unpacking tuples

# Tuples are made of several items just like a list, but they cannot be modified in any way. It is very common for tuples to be used to represent data from a database. If you have a tuple like ('chocolate chip cookies', 15) and you want to access each part of the data, you can use an index just like a list. However, you can also "unpack" the tuple into multiple variables such as type, count = ('chocolate chip cookies', 15) that will set type to 'chocolate chip cookies' and count to 15.

# Often you'll want to pair up multiple array data types. The zip() function does just that. It will return a list of tuples containing one element from each list passed into zip().

# When looping over a list, you can also track your position in the list by using the enumerate() function. The function returns the index of the list item you are currently on in the list and the list item itself.
# See below: 

# Pair up the boy and girl names: pairs
pairs = zip(girl_names, boy_names)

# Iterate over pairs
for idx, pair in enumerate(pairs):
    # Unpack pair: girl_name, boy_name
    girl_name, boy_name = pair
    # Print the rank and names associated with each rank
    print('Rank {}: {} and {}'.format(idx, girl_name, boy_name))

# "making tuples by accident" is the most stupid exercise in this course

##############################################
# Sets: unique in unordered, mutable
##############################################


# Sets tend to come from listss

set(list)
.add() # adds, BUT only if unique..i.e. not already in the set.
.update() # merges in another set or list 
.discard() # 
.pop() # remove AND return
.union() # combine sets TO FORM A SET ...ei duplikaatteja
.intersection() (overlapping in two)
.difference() # Target  important 


    
# Find the union: all_names
all_names = baby_names_2011.union(baby_names_2014)

# Print the count of names in all_names
print(len(all_names))

# Find the intersection: overlapping_names
overlapping_names = baby_names_2011.intersection(baby_names_2014)

# Print the count of names in overlapping_names
print(len(overlapping_names))

## another example
# Create the empty set: baby_names_2011
baby_names_2011 = set()

# Loop over records and add the names from 2011 to the baby_names_2011 set
for row in records:
    # Check if the first column is '2011'
    if row[0] == '2011':
        # Add the fourth column to the set
        baby_names_2011.add(row[3])

# Find the difference between 2011 and 2014: differences
differences = baby_names_2011.difference(baby_names_2014)

# Print the differences
print(differences)

#################################
# Dicts
#################################
Init: {}

released = {
		"iphone" : 2007,
		"iphone 3G" : 2008,
		"iphone 3GS" : 2009,
		"iphone 4" : 2010,
		"iphone 4S" : 2011,
		"iphone 5" : 2012
	}
print released

Print all key and values

for key,val in released.items():
    print key, "=>", val
    
# Add a value to the dictionary
# You can assign to an individual dictionary entry to add it or modify it

#the syntax is: mydict[key] = "value"
released["iphone 5S"] = 2013
print released

>>Output
{'iphone 5S': 2013, 'iphone 3G': 2008, 'iphone 4S': 2011, 'iphone 3GS': 2009,
'iphone': 2007, 'iphone 5': 2012, 'iphone 4': 2010}


# Names
for name in dict: print(name)

Safely finding by key: 
.get() allows to safely access key without thrown exception.
...it returns None

# ! This is good:
# Default value, in case not found, will return 'Not found', not exception, not None!
art_galleries.get('Louvre', 'Not found')

# Nested data - use for hierarchies
art_galleries.keys() -> dict_keys(['10027','..other zip code keys...'])
print(art_galleries['10027'])
{"Paige's Art Gallery": '(289 839-9843', 'Inner City Art Gallery Inc': '20983804-093'}
print(art_galleries['10027']['Inner City Art Gallery Inc'])
'20983804-093'

"Common way to deal with repeating data structures"

# another nice one

# Create an empty dictionary: names
names = dict()

# Loop over the girl names
for name, rank in female_baby_names_2012:
    # Add each name to the names dictionary using rank as the key
    names[rank] = name
    
# Sort the names list by rank in descending order and slice the first 10 items
for rank in sorted(names, key=names.get, reverse=True)[:10]:
    # Print each item
    print(names[rank])

# Safely print rank 7 from the names dictionary
print(names.get(7))

# Safely print the type of rank 100 from the names dictionary
print(type(names.get(100)))

# Safely print rank 105 from the names dictionary or 'Not Found'
print(names.get(105, 'Not Found'))

# ANOTHER EXAMPLE 

# Print a list of keys from the boy_names dictionary
print(boy_names.keys())

# Print a list of keys from the boy_names dictionary for the year 2013
print(boy_names[2013].keys())

# Loop over the dictionary
for year in boy_names:
    # Safely print the year and the third ranked name or 'Unknown'
    print(year, boy_names[2013].get(3, 'Unknown'))

# Updates:
.update()
galleries_lkj = [(your_stuff_here, and_here), (your_stuff_here, and_here)]
art_galleries['your_key'].update(galleries_lkj) # this updates the data behind that key


del # throws error
pop() # returns safely


####################
# example: 
# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
boy_names[2011] = names_2011

# Update the 2012 key in the boy_names dictionary
boy_names[2012].update([(1,'Casey'), (2,'Aiden')])

# Loop over the boy_names dictionary 
for year in boy_names:
    # Loop over and sort the data for each year by descending rank
    for rank in sorted(boy_names[year], reverse=True)[:1]:
        # Check that you have a rank
        if not rank:
            print(year, 'No Data Available')
        # Safely print the year and the least popular name or 'Not Available'
        print(year, boy_names[year].get(rank, 'Not Available'))
#

# Remove 2011 and store it: female_names_2011
female_names_2011 = female_names.pop(2011)

# Safely remove 2015 with an empty dictionary as the default: female_names_2015
female_names_2015 = female_names.pop(2015, {})

# Delete 2012
del female_names[2012]

# Print female_names
print(female_names)


# ITEROINTI

# Iterate over the 2014 nested dictionary
for rank, name in baby_names[2014].items():
    # Print rank and name
    print(rank,name)
    
# Iterate over the 2012 nested dictionary
for rank, name in baby_names[2012].items():
    # Print rank and name
    print(rank,name)

# in
# Check to see if 2011 is in baby_names
if 2011 in baby_names:
    # Print 'Found 2011'
    print('Found 2011')
    
# Check to see if rank 1 is in 2012
if 1 in baby_names[2012]:
    # Print 'Found Rank 1 in 2012' if found
    print('Found Rank 1 in 2012')
else:
    # Print 'Rank 1 missing from 2012' if not found
    print('Rank 1 missing from 2012')
    
# Check to see if Rank 5 is in 2013
if 5 in baby_names[2013]:
   # Print 'Found Rank 5'
   print('Found Rank 5')
 

# THE EXAMPLE IS WRONG!!! Script output is NOT IN ORDER
# Create an empty dictionary: names
# names = dict()

# # Loop over the girl names
# for name, rank in female_baby_names_2012:
    # # Add each name to the names dictionary using rank as the key
    # names[rank] = name
    
# # Sort the names list by rank in descending order and slice the first 10 items
# for rank in sorted(names, reverse=True)[:10]:
    # # Print each item
    # print(names[rank])
...this will not be in order but it is the "correct answer" - mine produced correct output and it was "wrong".

# ...oh the second one is wrong as well. Guys...


# this too is wrong, the initialized object is a tuple, not a dict: 

# Import the python CSV module
import csv

# Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv','r')

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary
    baby_names = baby_names.update( {row[5], row[3]} )
    # can't do this, is a tuple baby_names.update(row[5], row[3])

# Print the dictionary keys
print(baby_names[0])

# THE "CORRECT "ANSWER:
# Import the python CSV module
import csv

# Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary
    baby_names[row[5]] = row[3]

# Print the dictionary keys
print(baby_names.keys())


###### another....
# Import the python CSV module
import csv

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
csvfile = open('baby_names.csv','r')

# Loop over a DictReader on the file
for row in csv.DictReader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary: baby_names
    baby_names[row['RANK']] = row['NAME']

# Print the dictionary keys
print(baby_names.keys())

########################################
# Collections module
########################################

counter = dictionary used for counting data, measuring frequency - good stuff!

from collections import Counter
print(nyc_eatery_count_by_types['Restaurant'])
15

# .most_common() returns counter values in descending order 
# top 3
print(nyc_eatery_count_by_types.most_common(3))
[('Mobile Food Truck', 114), ('Food Cart', 74), ('Snack Bar', 24)]
# ...great for frequency analytics

## A COUNTER EXAMPLE
# Import the Counter object
from collections import Counter

# Print the first ten items from the stations list
print(stations[:10])

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Print the station_count
print(station_count)

## most common: 
# Import the Counter object
from collections import Counter

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Find the 5 most common elements
print(station_count.most_common(5))

from collections import defaultdict
eateries_by_park = defaultdict(list)
# defaultdict ja täydentäminen
for park_id, name in nyc_eateries_parks:
    if park_id not in eateries_by_park:
        eateries_by_park[park_id] = []
    eateries_by_park[park_id].append(name)
    
from collections import defaultdict
eatery_contact_types = defaultdict(int)
    for eatery in nyc_eateries
        if eatery.get('phone'):
            eatery_contact_types['phones'] += 1
        if eatery.get('website'):
            eatery_contact_types['websites'] += 1


# ANOTHER EXAMPLE - 

# correct answer: 
# Create an empty dictionary: ridership
ridership = {}

# Iterate over the entries
for date, stop, riders in entries:
    # Check to see if date is already in the dictionary
    if date not in ridership: #..but there can be nothing there!!! I emptied it. WTF?
        # Create an empty list for any missing date
        ridership[date] = []
    # Append the stop and riders as a tuple to the date keys list
    ridership[date].append((stop, riders))
    
# Print the ridership for '03/09/2016'
print(ridership['03/09/2016'])


## AND ANOTHER, THIS ONE USING DEFAULTDICT

# Import defaultdict
from collections import defaultdict

# Create a defaultdict with a default type of list: ridership
ridership = defaultdict(list)

# Iterate over the entries
for date, stop, riders in entries:
    # Use the stop as the key of ridership and append the riders to its value
    if stop not in ridership:
        ridership[stop] = []
    ridership[stop].append(riders)
    
# Print the first 10 items of the ridership dictionary
print(list(ridership.items())[:10])


###### ordering 
# yes, they are ordered, since 3.6 (2017)
from collections import OrderedDict
nyc_eatery_permits = OrderedDict()
for eatery in nyc_eateries:
    nyc_eatery_permits[eatery['end_date']] = eatery
    
print(list(nyc_eatery_permits.items())[:3]
('2029-04-28', {'name': 'Union Square Seasonal Cafe',
'location': 'Union Square Park', 'park_id': 'M089',
'start_date': '2014-04-29', 'end_date': '2029-04-28', 
'description': None, 'permit_number': 'M89-SB-R', 'phone': '212-677-7818', 
'website': 'http://www.thepavilionnyc.com/', 'type_name': 'Restaurant'})

# OrderedDict power feature:
# .popitem() method returns items in reverse insertion order
print(nyc_eatery_permits.popitem())
('2029-04-28', {'name': 'Union Square Seasonal Cafe',
'location': 'Union Square Park', 'park_id': 'M089',
'start_date': '2014-04-29', 'end_date': '2029-04-28',
'description': None, 'permit_number': 'M89-SB-R', 'phone': '212-677-7818',
'website': 'http://www.thepavilionnyc.com/', 'type_name': 'Restaurant'})

# You can use the last=False keyword argument to return the items in insertion order

###########
# Import OrderedDict from collections
from collections import OrderedDict

# Create an OrderedDict called: ridership_date
ridership_date = OrderedDict()

# Iterate over the entries
for date, riders in entries:
    # If a key does not exist in ridership_date, set it to 0
    if date not in ridership_date:
        ridership_date[date] = 0
        
    # Add riders to the date key in ridership_date
    ridership_date[date] += riders
    
# Print the first 31 records
print(list(ridership_date.items())[:31])


# Print the first key in ridership_date
print(list(ridership_date.keys())[0])

# Pop the first item from ridership_date and print it
print(ridership_date.popitem())

# Print the last key in ridership_date
print(list(ridership_date.keys())[-1])

# Pop the last item from ridership_date and print it
print(ridership_date.popitem(last=False))


#### namedtuple
alternative to pandas dataframe
from collections import namedtuple

from collections import namedtuple
Eatery = namedtuple('Eatery', ['name', 'location', 'park_id',
'type_name'])
eateries = []
for eatery in nyc_eateries:
    details = Eatery(eatery['name'],
                     eatery['location'],
                     eatery['park_id'],
                     eatery['type_name'])
    eateries.append(details)
   
   
print(eateries[0])
Eatery(name='Mapes Avenue Ballfields Mobile Food Truck',
location='Prospect Avenue, E. 181st Street',
park_id='X289', type_name='Mobile Food Truck'

for eatery in eateries[:3]:
    print(eatery.name)
    print(eatery.park_id)
    print(eatery.location)

### POPULATING NAMED TUPLE
# Import namedtuple from collections
from collections import namedtuple

# Create the namedtuple: DateDetails
DateDetails = namedtuple('DateDetails', ['date', 'stop', 'riders'])

# Create the empty list: labeled_entries
labeled_entries = []

# Iterate over the entries
for date, stop, riders in entries:
    # Append a new DateDetails namedtuple instance for each entry to labeled_entries
    labeled_entries.append(DateDetails(date, stop, riders))
    
# Print the first 5 items in labeled_entries
print(labeled_entries[:5])


# Iterate over the first twenty items in labeled_entries
for item in labeled_entries[:20]:
    # Print each item's stop
    print(item.stop)

    # Print each item's date
    print(item.date)

    # Print each item's riders
    print(item.riders)

############ NUMPY ARRAY SLICING RECAP
slice = start:stop:stride
no start e.g. [:20] = from start of array 
no stop e.g. [20:] = implicitly at end 
missing stride = stride = 1

1d array: a[slice], 2d array A[slice0, slice1]
negative -1 ..from end or backwards - 
