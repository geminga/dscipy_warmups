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

