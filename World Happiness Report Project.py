#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[3]:


Report=pd.read_csv('happiness_index_report.csv')
Report


# In[4]:


Report.keys()


# In[9]:


df=pd.DataFrame(Report)
df


# In[12]:


df.shape


# In[13]:


df.columns


# In[14]:


df.dtypes


# In[20]:


sns.heatmap(df.corr(),annot=True)


# In[15]:


df.isnull().sum()


# In[16]:


sns.heatmap(df.isnull())


# In[17]:


df.describe()


# High Standard Deviation is present in Happiness Rank,means data is spread too much.Range is high
# 

# In[18]:


df.skew()


# In[19]:


df['Happiness Rank'].plot.box()


# In[21]:


df.plot(kind='box',subplots=True,layout=(2,7))


# In[23]:


df['Trust (Government Corruption)'].plot.box()


# In[24]:


df['Standard Error'].plot.box()


# In[25]:


df['Generosity'].plot.box()


# In[26]:


df['Dystopia Residual'].plot.box()


# In[27]:


df['Freedom'].plot.box()


# In[28]:


df['Family'].plot.box()


# In[30]:


df['Health (Life Expectancy)'].plot.box()


# In[35]:


#removing outliers
from scipy.stats import zscore
z=np(zscore(df))
z


# In[ ]:




