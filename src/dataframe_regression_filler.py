
def dataframe_regression_filler(dataframe, *args, column):
"""Fills a dataframe column with regression model predictions. 
Inputs: dataframe (containing both the predicting variables and the predicted variable), 
columns of said dataframe used to build predictive regression model, 
column of said dataframe to fill with products of said regression model.

..returns a dataframe, with values filled.

"""
# Rather import these in the program itself?
import pandas as pd
import statsmodels.api as sm

    try:
        predictors = pd.DataFrame(dataframe, columns = [*args])
    except:
        print('You sure you are predicting with columns that actually exist in the dataframe? Here are the columns in the dataframe')
        list(dataframe) # if does not work as expected, just embed the list(df) in a print()
    target_column = column
    dataframe_to_be_filled = dataframe

    model = sm.OLS(target_column,predictors).fit()
    # .... what here?
        # Should I handle the whole dataframe or just a subset of vars relevant to the exercise 
            # Subset
        # should I overwrite the dataframe inplace - or create a copy?
            # Make a copy and handle that
            # Actually NumPy:s handled often
        # how do I overwrite the dataframe - does the "inplace" really work?
            # 1) value X := Y in the original one (inplace)
            # 2) -> to the new dataframe via a calculated NumPy array -> calculate a new column with the same name with the new values 
        
            # how do I overwrite the dataframe - does the "inplace" really work?
    # predictions = model.predict(X)
    
    DataFrame.replace(to_replace=target_column, value=model.OUTPUT, inplace=True, limit=None, regex=False, method='pad')
    
    return filled_dataframe # this actually will be a new one, calculated on the basis of the old.
