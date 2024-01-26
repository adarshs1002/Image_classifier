#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# using pandas to read the database in the folder
data = pd.read_csv('mnist_train.csv')


# In[3]:


#To view column heads
data.head()


# In[4]:


#extracting data from the dataset and viewing them up close
a = data.iloc[3,1:].values


# In[5]:


#reshaping the extracted data into a reasonable size
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[6]:


# preparing the data 
# seperating labels and data values
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[7]:


#creating test and train sizes/ batches
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[8]:


#check data
x_train.head()


# In[9]:


y_train.head()


# In[10]:


#call rf classifier
rf = RandomForestClassifier(n_estimators=100)


# In[11]:


#fit the model
rf.fit(x_train, y_train)


# In[12]:


#prediction on test data
pred = rf.predict(x_test)


# In[13]:


pred


# In[14]:


# check accuracy of prediction
s = y_test.values

#calculate the number predictions that are correct
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count + 1
        


# In[15]:


print(count)
print("The total number of values that the prediction code run on is ", len(pred))
print("The percent of accuracy is :",int(count)*100/int(len(pred)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




