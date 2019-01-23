# silly func
def squared_4():
    new_value = 4 ** 2
    print(new_value)

# Slightly more useful func
def square(input_value):
    new_value = input_value ** 2
    print(new_value)
    
# A returning func
def return_square(input_value):
    """Returns a square of a value"""
    new_value = input_value ** 2
    return new_value
    
print(return_square(15))

def raise_to_power(input_number, power_to_raise_to):
    """Raise parameter 1 to power given in parameter 2"""
    new_value = input_number ** power_to_raise_to
    return new_value
    
print(raise_to_power(2, 4))


# Tuples
# ..a reminder: you cant modify tuples once constructed. (I think the same was with a string)

def raise_both_parameters_to_power_of_each_other(value1, value2):
    """Raise parameter 1 to power given in parameter 2 AND vice versa. Return tuple with results"""
    new_value1 = value1 ** value2
    new_value2 = value2 ** value1
    
    tuple_of_results = (new_value1, new_value2)
    
    return tuple_of_results
    
    
# Multiple val returning functions
# Tuple can be created just by feeding comma separated values to a var.
# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = shout1, shout2

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)

# RECAP - own funcs: remember 
# func header, 
    # Body, with 
    # Docstring with """
    # returning the computation result


# DATAFRAMES TO USE
# Example with pre-loads 
# Twitter data in csv : 'tweets.csv' file has been imported into the tweets_df 
# pandas has been imported as pd
# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result
result = count_entries(tweets_df, 'lang')

# Print the result
print(result)



# SCOPE AND USER-DEFINED FUNCTIONS
# (
# Global, in the main body; 
# local - defined inside a function during execution, gone when run 
# built in: pre-defined
# )

# search of var vals: local, if not found, global, if not found, built-in.
