#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


data = pd.read_csv('C:\\Users\\prane\\Documents\\Univ_of_Bristol\\mini-project\\backup-file\\fake_transactional_data_24.csv')


# In[6]:


data.shape


# In[ ]:





# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.nunique()


# In[13]:


# Check for null values
print(data['from_totally_fake_account'].isnull().sum())
print(data['monopoly_money_amount'].isnull().sum())
print(data['to_randomly_generated_account'].isnull().sum())
print(data['not_happened_yet_date'].isnull().sum())


# In[28]:


# Check for non-float values in from_totally_fake_account' and 'monopoly_money_amount'
non_float_values = data['from_totally_fake_account'][~data['from_totally_fake_account'].astype(str).str.replace('.', '', 1).str.isdigit()]
non_float_values.append(data['monopoly_money_amount'][~data['monopoly_money_amount'].astype(str).str.replace('.', '', 1).str.isdigit()])

# Print the result
if not non_float_values.empty:
    print("Found non-float values in float columns 'from_totally_fake_account', 'monopoly_money_amount':")
    print(non_float_values)
else:
    print("No non-float values found in float columns 'from_totally_fake_account', 'monopoly_money_amount'.")


# In[29]:


# Check for non-date values in feature not_happened_yet_date

try:
    pd.to_datetime(data['not_happened_yet_date'], errors='raise')
    print("No unexpected values found in 'not_happened_yet_date'.")
except ValueError as e:
    print(f"Unexpected values found in 'not_happened_yet_date': {e}")


# In[61]:


# The feature'to_randomly_generated_account' is a mix of str and int/float
import re

# Validation for string values : check if empty or special characters are present
str_validation = data['to_randomly_generated_account'][~data['to_randomly_generated_account'].astype(str).str.match('^[a-zA-Z0-9_]*$')]
if not str_validation.empty:
    print("Blank values or special characters found in 'to_randomly_generated_account':")
    print(str_validation.unique())
    print(len(str_validation))
else:
    print("No blank values or special characters found in 'to_randomly_generated_account'.")


# In[66]:


data['from_totally_fake_account'].apply(lambda x: len(str(x))).unique()


# In[79]:


#data['to_randomly_generated_account'][data['to_randomly_generated_account'].astype(str).str.match('^[0-9]') & data['to_randomly_generated_account'].apply(lambda x: len(str(x)) < 5)].unique()

data['to_randomly_generated_account'][data['to_randomly_generated_account'].astype(str).str.match('^[0-9]')].groupby(data['to_randomly_generated_account'].apply(lambda x: len(str(x)))).size().reset_index(name='count')


# In[4]:


data['to_randomly_generated_account'][data['to_randomly_generated_account'].astype(str).str.match('^[a-zA-Z_&]')].unique()


# In[18]:


filtered_rows = data['to_randomly_generated_account'][data['to_randomly_generated_account'].astype(str).str.match('^[0-9]')]
len(filtered_rows)


# In[ ]:




