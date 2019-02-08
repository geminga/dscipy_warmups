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


# Functions with one default arg
# NOTE 
# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
     exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo("Hey")

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo("Hey",5)


# mixing positional and by name args usage with default values.
# Define shout_echo
def shout_echo(word1, echo=1, intense = False):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Capitalize echo_word if intense is True
    if intense is True:
        # Capitalize and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo("Hey",5,True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo("Hey",intense=True)

# Print values
print(with_big_echo)
print(big_no_echo)

# Variable amount of input vars - a set in the example.

# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = str()

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)
# luke
# lukeleiahanobidarth

# KWARGS
# within the function definition, kwargs is a dictionary.

# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name="luke", affiliation="jedi", status="missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")


# NO JOO, TÄSSÄ ESIMERKKI, JOSSA *ARGS
# Define count_entries()
# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    #Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
    
        # Iterate over the column in DataFrame
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)