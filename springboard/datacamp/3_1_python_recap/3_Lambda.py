##############################################################
# LAMBDAS
##############################################################

# Example:
raise_to_power = lambda x, y: x**y 
# params,then what you do with them AND what it returns. 
# Note how much space is saved.

nums = [48,6,9,21,1}

##############################################################
# MAP 
##############################################################
# Useful, with e.g. map(function, sequence)
square_all = map(lambda num: num ** 2, nums)
# so, map does a for each with a function and what ever you gave.
# Here, lambda (raise to power of 2 as lambda) takes in num as its only parameter and then for that a power of 2, sequence fed to this is nums.

# Output out: 
print(list(square_all))

# simple lambda:
# add_bangs adds '!!!' to the end of a single parameter 
add_bangs = lambda a: a + '!!!'
# recommended format: 
add_bangs = (lambda a: a + '!!!')

def echo_word(word1, echo):
    """Concatenate echo copies of word1."""
    words = word1 * echo
    return words
    
# ..to lambda 
echo_word = (lambda word1, echo: word1 * echo)

# then example with map(function, iterable) applies a function over an object, such as a list...so e.g. a lambda-function..
# ...becomes sturdy with lambdas, since without lambda, the defs put in map might end up taxing the working memory of the programmer. Now fits on one line sometimes maybe.


## A BASIC EXAMPLE
# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda item: item + '!!!', spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list = list(shout_spells)

# Convert shout_spells into a list and print it
print(shout_spells_list)

##############################################################
# using lambda with FILTER (returns BOOLEAN)
##############################################################
# generally 
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)

# Output: [-5, -4, -3, -2, -1]

# So, in short, how to cut some data out based on length.
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member) > 6, fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Convert result into a list and print it
print(result_list)


##############################################################
# REDUCE - returns one item
##############################################################

# Import reduce from functools
from functools import reduce 

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1, item2: item1 + item2, stark)

# Print the result
print(result)
## 3n t4y51n t4junnu+ m1k51 +u733 yk51 +u705