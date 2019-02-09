
def dataframe_regression_filler(dataframe, *args, column):
"""Fills a dataframe column with regression model predictions. 
Inputs: dataframe (containing both the predicting variables and the predicted variable), 
columns of said dataframe used to build predictive regression model, 
column of said dataframe to fill with products of said regression model.

..returns a dataframe, with values filled.

"""
import pandas as pd
import statsmodels.api as sm

    predictors = pd.DataFrame(dataframe, columns = [*args])
    target_column = column
    dataframe_to_be_filled = dataframe

    model = sm.OLS(target_column,predictors).fit()
    # .... what here? - how do I overwrite the dataframe?
    # predictions = model.predict(X)
    
    DataFrame.replace(to_replace=target_column, value=model.OUTPUT, inplace=True, limit=None, regex=False, method='pad')
    
    return filled_dataframe 
