
# coding: utf-8

# In[1]:


# PYTHON DATA ACCESS DEMO


# In[1]:


import sys
import pandas as pd
import csv
import logging
import sklearn
import numpy
import scipy
import matplotlib
import boto3
import ckanapi
import os


# In[2]:


os.environ["AWS_ACCESS_KEY_ID"] = OMITTED!!
os.environ["AWS_SECRET_ACCESS_KEY"] = OMITTED!!


# In[3]:


sys.path.insert(0,"C:\\Users\\andro\\repositories\\sdh-alfa\\src\\lib\\scratch\\karri\\smartdatahub_py")


# In[4]:


import smartdatahub


# In[8]:


pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 5000)


# In[5]:


pnp = smartdatahub.dataset(LogLevel="INFO", Url="http://store.smartdatahub.io/dataset/fi_sdh_postalcode_profile_finland/resource/73772aa9-72f9-4dbb-8217-3e19ecf8248e")


# In[6]:


# returns a pandas dataframe
pnp.get_data()


# In[9]:


print(type(pnp))


# In[10]:


pd_df=pnp.data


# In[11]:


print(type(pd_df))


# In[12]:


list(pd_df)


# In[13]:


list(pd_df.index)


# In[26]:


pd_df.loc[:,'orig_extractdate']


# In[14]:


pd_df.loc[:,:]


# In[15]:


# OK, now we are talking. 
# First, drop the geom polygons, they eat space and we won't calculate with them now.
pd_df.drop(columns=['geom_geojson', 'geom_geotext','geom_geojson_sea',
 'geom_geotext_sea'])


# In[16]:


# let's select inhabitants and amount of schools 
df_inhabitants_schools = pd_df[['inhabitants_total','schools_basic_upper_secondary']].copy()


# In[17]:


df_inhabitants_schools


# In[33]:


df_inhabitants_schools.corr(method='spearman')


# In[39]:


df_inhabitants_schools.corr(method='spearman')


# In[19]:


df_children_7_12_schools = pd_df[['inhabitants_age_7_12','inhabitants_proportional_age_7_12','schools_basic_upper_secondary']].copy()


# In[20]:


df_children_7_12_schools.corr(method='kendall')


# In[21]:


pd_df.corr(method='kendall')


# In[22]:


correlation_kendall_to_file = pd_df.corr(method='kendall')


# In[23]:


print(type(correlation_kendall_to_file))


# In[26]:


correlation_kendall_to_file.to_csv("C:\\Users\\andro\\repositories\\sdh-alfa\\src\\lib\\scratch\\manne\\kendall_correlations_to_file.csv")


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(correlation_kendall_to_file, annot=True, fmt="d", linewidths=.5, ax=ax)
sns.heatmap(correlation_kendall_to_file, annot=True, fmt="f", linewidths=.5, ax=ax).show()


# In[31]:


f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(correlation_kendall_to_file, annot=True, fmt="d", linewidths=.5, ax=ax)
sns.heatmap(correlation_kendall_to_file)

