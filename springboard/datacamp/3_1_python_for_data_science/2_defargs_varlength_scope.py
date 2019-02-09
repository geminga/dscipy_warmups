# SCOPE
# In this exercise, you will practice what you've learned about scope in functions. The variable num has been predefined as 5, alongside the following function definitions:

def func1():
    num = 3
    print(num)

def func2():
    global num
    double_num = num * 2
    num = 6
    print(double_num)

# Try calling func1() and func2() in the shell, then answer the following questions:

    # What are the values printed out when you call func1() and func2()?
    # What is the value of num in the global scope after calling func1() and func2()?

# First of all, using these within print causes nasty behaviour, a "None" is printed, lack of type, don't know really.

# Calling them instead:
# func1 
    # prints the local var (assigned a 3)
# func2 
    # prints what global num (5 at the time) would be when used in double_num multiplied by 2. 
    # However....after that it is assigned a 6, so the global num is now 6
    

# EXERCISE:
# Create a string: team
team = "teen titans"

# Define change_team()
def change_team():
    """Change the value of the global variable team."""

    # Use team in global scope
    global team

    # Change the value of team in global: team
    team = "justice league"
# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)    

# gives first teen titans and then justice league


# ...whaaaaaattt builtins is not built in? wut?

In [1]: import builtins
In [2]: dir(builtins)

# WOW! I actually had no idea that errors are in builtins and NEED TO BE IMPORTED?!?
# ..and getipython is in "builtins"...interesting.

# Var assignments - unless you say Global or Nonlocal, of course is only local in scope.

# DEFAULT ARGS
# well, args(a_string, a_num=1) is pretty obvious, but, take all args is neater!
def add_all(*args):
    """Sum all values together"""
    
    #init
    sum_all = 0
    
    # Accumulate sum
    for num in args:
        sum_all += num
        
    return sum_all
    
def take_in_all_key_word_pair_arguments(**kwargs):
    """Print out key-value pairs in **kwargs"""
    
    # Print them!
    for key, value in kwargs.items():
        print(key + ": " + value)
        

