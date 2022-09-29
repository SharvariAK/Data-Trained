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


# In[2]:


Report=pd.read_csv('happiness_index_report.csv')
Report


# In[3]:


Report.keys()


# In[4]:


df=pd.DataFrame(Report)
df


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


sns.heatmap(df.corr(),annot=True)


# In[9]:


df.isnull().sum()


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


df.describe()


# High Standard Deviation is present in Happiness Rank,means data is spread too much.Range is high
# 

# In[12]:


df.skew()


# In[13]:


df['Happiness Rank'].plot.box()


# In[14]:


df.plot(kind='box',subplots=True,layout=(2,7))


# In[15]:


df['Trust (Government Corruption)'].plot.box()


# In[16]:


df['Standard Error'].plot.box()


# In[17]:


df['Generosity'].plot.box()


# In[18]:


df['Dystopia Residual'].plot.box()


# In[19]:


df['Freedom'].plot.box()


# In[20]:


df['Family'].plot.box()


# In[21]:


df['Health (Life Expectancy)'].plot.box()


# In[22]:


#AS only this parameters may affect the results
#independent variables
x=df.iloc[:,4:13]
x


# In[23]:


#dependent variables
y=df.iloc[:,-9]
y


# In[24]:


x.shape


# In[25]:


y.shape


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=45)


# In[27]:


x_train.shape


# In[28]:


x_test.shape


# In[29]:


y_train.shape


# In[30]:


y_test.shape


# In[31]:


lm=LinearRegression()


# In[32]:


lm.fit(x_train,y_train)


# In[33]:


lm.coef_


# In[34]:


lm.intercept_


# In[35]:


df.columns


# In[36]:


lm.score(x_train,y_train)


# In[37]:


#predict the value
pred=lm.predict(x_test)
print('Predicted result:',pred)
print('actual result',y_test)


# In[38]:


print('error')
print('Mean absolute error:',mean_absolute_error(y_test,pred))
print('Mean squared error:',mean_squared_error(y_test,pred))
print(' Root Mean squared error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[39]:


#r2 score means change coming in y whenever x is being changed.
from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[40]:


p=np.array([0.03328,1.32548,1.36058,0.87464,0.64938,0.48357,0.34139,2.49204])


# In[41]:


p.shape


# In[42]:


p=p.reshape(1,-1)


# In[43]:


p.shape


# In[44]:


lm.predict(p)


# In[ ]:




