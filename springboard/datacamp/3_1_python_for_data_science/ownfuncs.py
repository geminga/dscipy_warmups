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
# You cant modify tuples once constructed

def raise_both_parameters_to_power_of_each_other(value1, value2):
    """Raise parameter 1 to power given in parameter 2 AND vice versa. Return tuple with results"""
    new_value1 = value1 ** value2
    new_value2 = value2 ** value1
    
    tuple_of_results = (new_value1, new_value2)
    
    return tuple_of_results
    




    

    