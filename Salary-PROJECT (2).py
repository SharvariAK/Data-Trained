#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as st
from sklearn.metrics import mean_squared_error,mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# In[3]:


dataset=pd.read_csv('Salary.csv')


# In[4]:


dataset


# In[5]:


dataset.keys()


# In[6]:


df=pd.DataFrame(dataset)
df


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[13]:


sns.heatmap(df.corr(),annot=True)


# In[14]:


df.skew()


# In[15]:


df['yrs.since.phd'].plot.hist()


# In[16]:


x=dataset.iloc[:,0:-1]
x


# In[17]:


y=dataset.iloc[:,-1]
y


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=45)


# In[19]:


x_train.shape


# In[20]:


x_test.shape


# In[21]:


y_train.shape


# In[22]:


y_test.shape


# In[23]:


df['yrs.since.phd'].plot.box()


# In[24]:


df['yrs.service'].plot.box()


# In[25]:


df.plot(kind='box',subplots=True,layout=(2,7))


# In[26]:


lm=LinearRegression()


# In[27]:


lm.fit(x_train,y_train)


# # how to solve this error

# In[28]:


y_pred=lm.predict(x_test)
y_pred


# In[ ]:


y_test


# In[ ]:


plt.scatter(x,y,color='r')


# In[ ]:




