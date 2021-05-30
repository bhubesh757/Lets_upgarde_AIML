#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# pickle is used to save the model created by us


# In[3]:


df = pd.read_csv('hiring.csv')
df.head()


# In[4]:


df.isna().sum()


# In[5]:


# experience column

df['experience'].fillna(0 , inplace=True)


# In[6]:


df.isna().sum()


# In[7]:


df['test_score'].mean()


# In[8]:


df['test_score'].fillna(df['test_score'].mean() , inplace = True)


# In[9]:


df.isna().sum()


# In[10]:


# Dataset is cleaned 


# In[11]:


X = df.iloc[:,:-1]
X.head()


# In[12]:


X.shape


# In[13]:


X.experience


# In[14]:


#convert it text in the cols to the values


# In[15]:


# Convert text in the cols to integer values

def conv(x):
    dict = {'two':2, 'three':3, 'five':5, 'seven':7, 'ten':10, 0:0, 'eleven':11 }
    return dict[x]


# In[16]:


X['experience'] = X['experience'].apply(lambda x: conv(x))


# In[17]:


X.head()


# In[19]:


X.info()


# In[20]:


y = df.iloc[:,-1]
y


# In[22]:


# Modelling

from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[23]:


# fit the model
lr.fit(X,y)


# In[ ]:





# In[25]:


# prediction of th lr
y_pred = lr.predict(X)
y_pred


# In[26]:


y


# In[30]:


from sklearn.metrics import r2_score
r2_score(y_pred , y)


# In[32]:


X


# In[34]:


lr.predict([[3,9,7]])


# In[35]:


lr.predict([[10,10,10]])


# # Model Deployment

# In[36]:


import pickle

pickle.dump(lr , open('model.py' , 'wb'))
# dump this model by the name as name.py in the system


# In[37]:


model2 = pickle.load(open('model.py' ,'rb'))


# In[38]:


model2.predict([[3,9,7]])


# In[39]:


model2.predict([[10,10,10]])


# # Happy Learning Bhubesh SR

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




