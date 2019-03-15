import pandas as pd
df = your_dataset_to_dataframe

df.head()
df.tail()
df.info()

## Some types:
# Import numpy
import numpy as np

# Create array of DataFrame values: np_vals
np_vals = df.values

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]

np_vals = numpy.ndarray
np_vals_log10 = numpy.ndarray
df = pandas.core.frame.DataFrame

# read: 
pd.read_csv

# Dataframes from dict - first column -> index 
Katopa ne videot uudestaan ja plokkaa kaikki mitä se näyttää.
