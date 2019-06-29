#!/usr/bin/env python
# coding: utf-8

# In[11]:

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,cross_validate,train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer, RobustScaler
from sklearn.metrics import *
# import seaborn as sns
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys
import pickle
import os


# In[ ]:


a = sys.stdin.read()


# In[45]:


data_path, model_path = a.split()


# In[46]:


df = pd.read_csv("Data/HistoricalQuotesonemonth.csv",thousands=',')


# In[48]:


df.describe().T


# In[49]:


# df.head()


# In[50]:


test_set = df.drop('date',axis =1)


# In[51]:


test_set.head()


# In[52]:


y_test = test_set['open']
test_set.drop('open',axis=1,inplace=True)


# In[53]:


with open(model_path,'rb') as f:
    model = pickle.load(f)


# In[54]:


pred = model.predict(test_set)


# In[55]:


err_mae = mean_absolute_error(y_test,pred)
accuracy = r2_score(y_test,pred)


# In[56]:


print("error mae = {}, accuracy = {}".format(err_mae,accuracy))


# In[57]:


df['pred']=pred


# In[58]:


df.head()


# In[9]:


path = "predictions/pred.csv"


# In[60]:


df.to_csv(path)


# In[10]:


try:
    sys.stdout.write(path)
except Exception as e:
    print("Cannot return the path beacause ",e)
