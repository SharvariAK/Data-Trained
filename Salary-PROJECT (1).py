#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[24]:


dataset=pd.read_csv('Salary.csv')


# In[25]:


dataset


# In[26]:


dataset.keys()


# In[33]:


df=pd.DataFrame(dataset)
df


# In[34]:


df.shape


# In[35]:


df.dtypes


# In[36]:


df.describe()


# In[37]:


df.isnull().sum()


# In[39]:


sns.heatmap(df.corr(),annot=True)


# In[46]:


x=dataset.iloc[:,0:-1]
x


# In[47]:


y=dataset.iloc[:,-1]
y


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=45)


# In[49]:


x_train.shape


# In[50]:


x_test.shape


# In[51]:


y_train.shape


# In[52]:


y_test.shape


# In[53]:


df.skew() 


# In[54]:


df['yrs.since.phd'].plot.box()


# In[55]:


df['yrs.service'].plot.box()


# In[56]:


df.plot(kind='box',subplots=True,layout=(2,7))


# In[57]:


#removing outliers
from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[ ]:




