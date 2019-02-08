# Lambdas
# Example:
raise_to_power = lambda x, y: x**y 
# params,then what you do with them AND what it returns. 
# Note how much space is saved.

nums = [48,6,9,21,1}
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

# then example with map
# map(function, iterable) applies a function over an object, such as a list.
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
