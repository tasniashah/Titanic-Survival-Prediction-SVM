#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv("D:/titanic_clean.csv")


# In[3]:


data.dtypes


# In[4]:


data.columns


# In[5]:


data.shape


# In[6]:


data.size


# In[7]:


data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked', 'Title',
'GrpSize', 'FareCat', 'AgeCat'])


# In[8]:


X = data.drop(['Survived','PassengerId'],axis=1)
Y = data['Survived']
x_train,X_test,y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=91,shuffle=True)


# In[9]:


svm_model = SVC(kernel='linear')


# In[10]:


svm_model.fit(x_train, y_train)


# In[11]:


y_pred = svm_model.predict(X_test)


# In[12]:


accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

