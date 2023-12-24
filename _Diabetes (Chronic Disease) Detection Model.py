#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Making the Diabetic Detection Model using (Support Vector Machine --> Supervised Machine Learning)


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[4]:


dataset = pd.read_csv('diabetes.csv')


# In[5]:


dataset.info


# In[6]:


# Extracting some of the Statical measure of this data type


# In[7]:


dataset.describe()


# In[8]:


dataset['Outcome'].value_counts()


# In[9]:


dataset.groupby('Outcome').mean()


# In[10]:


# from the above observation we can see that people who are not diabetic have less glucose level
# and we can also see that people who are diabetic have higher mean glucose level


# In[11]:


# seperating data and labels

X = dataset.drop(columns = 'Outcome', axis = 1)
Y = dataset['Outcome']


# In[12]:


X 


# In[13]:


Y.shape


# In[14]:


# Standardising the Data - 


# In[15]:


scalar = StandardScaler()


# In[16]:


scalar.fit(X)


# In[17]:


Standardized_data = scalar.transform(X)


# In[18]:


print(Standardized_data)


# In[19]:


X = Standardized_data
Y = dataset['Outcome']


# In[20]:


# Splitting the Data into Training Data and Testing Data ( by Train_Test_split)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size= 0.2, stratify = Y, random_state=2)


# In[ ]:





# In[21]:


classifier = svm.SVC(kernel = 'linear')


# In[22]:


#training the support vetor Machine classifier (SVM Machine)

classifier.fit(X_train,Y_train)


# In[23]:


# Chekcing the predction of the model (Model Evaluation)

# Accuray Score on the training data

X_train_predction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predction, Y_train)


# In[24]:


print('Accuracy score of the training data : ' , training_data_accuracy)


# In[25]:


# finding the accuracy on test data


# In[26]:


X_train_predction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_train_predction, Y_test)


# In[27]:


print('Accuracy score of test data : ', test_data_accuracy)


# In[28]:


# Making a predictive system with input data


# In[29]:


input_data = (4,110,92,0,0,37.6,0.191,30)


# In[30]:


# changing the input data to numpy array


# In[31]:


input_data_as_numpy_array = np.asarray(input_data)


# In[32]:


#reshaping the array, as we are prdiction for one instance - (reason) we need the predction for just one data


# In[33]:


input_data_reshape = input_data_as_numpy_array.reshape(1,-1)


# In[34]:


# we cannot feed the data as such what has been given in input command because we have standardise the data
# We have standardised our training data

# Now we are Standardising our input data


# In[35]:


std_data = scalar.transform(input_data_reshape)


# In[36]:


print(std_data)


# In[37]:


prediction = classifier.predict(std_data)


# In[38]:


print(prediction)


# In[43]:


if (prediction[0] == 0):
    print(' The person is not diabetic')
else:
    print('The person is diabetic')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




