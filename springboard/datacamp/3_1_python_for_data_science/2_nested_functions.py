# Nested functions, also returning functions 
# Hm. Haven't come across use cases where returning functions as if from a factory would be needed..but maybe I will one day. Cool.

# ...but, the example is compelling.
def raise_val(n):
    """Returns an inner function, which then raises to any power you like"""
    
    def inner(x):
    """Raise x to the power of n."""
        raised = x ** n
        return raised
        
    return inner
    
# ..now, usage..you have a power raising function factory, from which you get quite usable functions as a result.
square = raise_val(2)
cube = raise_val(3)
print(square(2), cube(4))

# Let that sink in. I suddenly can see quite a bit of use cases here.

# Nonlocal...need for usage of that I didn't get. I will.

# Well, for the record.
# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))

# A mouthful: 
# closure  means that the nested or inner function remembers the state of its enclosing scope when called. Thus, anything defined locally in the enclosing scope is available to the inner function even when the outer function has finished execution.

# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))

# So, if a var has been mucked with in outer ranges, of course it reflects on any of its usage in inner ranges.
# ..and you can alter it like this e.g.:
# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word = word + word
    
    # Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        # Use echo_word in nonlocal scope
        nonlocal echo_word
        
        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word + '!!!'
    
    # Call function shout()
    shout()
    
    # Print echo_word
    print(echo_word)

# Call function echo_shout() with argument 'hello'
echo_shout('hello')
