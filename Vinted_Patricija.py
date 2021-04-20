#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages

import pandas as pd
import datetime
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot
import pickle


# In[2]:


# Load the training and test datasets

parq_file = 'data.parquet'
data = pd.read_parquet(parq_file, 'auto')

parq_file_test = 'test.parquet'
data_test = pd.read_parquet(parq_file_test, 'auto')


# In[3]:


# Training dataset has 666814 rows and 76 columns

data.shape


# In[4]:


# Test dataset has 74091 rows and 74 columns, 2 columns less

data_test.shape


# In[5]:


# Columns not present in the test set will be removed before training the model:
# listing_price_local, sale_time

for i in data.columns:
    if i not in data_test.columns:
        print(i)


# In[6]:


# Check column names, data types, non-null counts

data.info()
data_test.info()


# In[7]:


# Check format of sale_time and local_time to calculate the difference

data['sale_time']


# In[8]:


data['local_time']


# In[9]:


# Replace all None values in sale_time with a placeholder

data["sale_time"].replace({None: "2025-01-01T00:00:00.000-07:00"}, inplace=True)
data["sale_time"]


# In[10]:


# Replace all missing values with 'na' which will be encoded

data = data.fillna('na')

# Create a no_hashtags column containing a number of hashtags for that listing 
# which will be used as a variable instead of the hashtags column (easier to measure)

def no_hashtags(hashtags):
    no_hashtags = int(hashtags.size)
    return no_hashtags

data['no_hashtags'] = data.apply(lambda row: no_hashtags(row['hashtags']), axis=1)

# Add a column that contains the time in which the product was sold
# We only care about whether something was sold in 24 hours or not, so the placeholder dates
# will be a lot larger than that

def conversion(sale, local):
    frmt = "%Y-%m-%dT%H:%M:%S.%f%z"
    sold_in = datetime.strptime(sale, frmt) - datetime.strptime(local, frmt)
    return sold_in

data['sold_in'] = data.apply(lambda row: conversion(row['sale_time'], row['local_time']), axis=1)


# In[11]:


# Check if the new column generated correctly

data['sold_in']


# In[13]:


# Create a column separating listings sold in 24 hours (1) and ones that weren't (0)
# This will be the dependent variable

def separation(sold_in):
    if sold_in <= timedelta(days = 1):
        sold_in_24 = 1
    elif sold_in > timedelta(days = 1):
        sold_in_24 = 0
    return sold_in_24

data['sold_in_24'] = data.apply(lambda row: separation(row['sold_in']), axis=1)

# Check if the calculations were correct

data2 = data[data['sold_in_24'] == 1]
data3 = data2.filter(['local_time', 'sale_time', 'sold_in','sold_in_24'])
data3[0:10]


# In[ ]:


# See all columns and what their values are to determine which columns should already be dropped

pd.set_option('display.max_columns', None)


# In[ ]:


# data[0:10]


# In[14]:


# Drop columns that should not contribute to success of a listing, to ease the training process
# Listing_price_local is removed because test set doesn't have this column

data = data.drop(['title','listing_price_local', 'portal','sold_in', 'brand_is_verified', 'hashtags', 'local_time', 'sale_time'], axis = 1)
data = data.drop(data.loc[:,'custom_shipping_price_domestic':'disposal_conditions'].columns, axis = 1)
data = data.drop(data.loc[:, 'basic_verification_local_time':'lifetime'].columns, axis = 1)
data = data.drop(data.loc[:, 'phone_verification_local_time':'second_sale_local_time'].columns, axis = 1)
data = data.drop(data.loc[:, 'registration_app_id':'registration_type'].columns, axis = 1)

# Change the boolean "True" or "False" column into 1 and 0
        
data['with_video'] = data['with_video']*1

data.info()


# In[15]:


# Turn object data type columns into categories and then encode each category 
# so all variables are numeric

for col_name in data.columns:
    if(data[col_name].dtype == 'object'):
        data[col_name]= data[col_name].astype('category')
        data[col_name] = data[col_name].cat.codes


# In[16]:


# Check how the dataset looks now
# Only float and integer columns left which the model will be able to handle

data.info()


# In[17]:


# Assign independent (X) and dependent (y) variables

y = data.pop('sold_in_24')
X = data


# In[18]:


# Split the data into training, test and validation sets

seed = 50

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = seed)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state = seed)


# In[19]:


# Create and fit Random Forest model

model = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = seed)

model.fit(X_train,y_train)


# In[20]:


# Make predictions on the validation set

y_pred = model.predict(X_val)


# In[22]:


# Model evaluation / Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_m_val = confusion_matrix(y_val, y_pred)
confusion_m_val


# In[23]:


# Accuracy, Precision, Recall and F1 scores


accuracy_val = accuracy_score(y_val, y_pred)
print("accuracy:", accuracy_val)

precision_val = precision_score(y_val, y_pred)
print("precision:", precision_val)

recall_val = recall_score(y_val, y_pred)
print("recall:", recall_val)

f1_val = f1_score(y_val, y_pred)
print("f1:", f1_val)


# In[24]:


# Save this model

pickle.dump(model, open('baseline_model.pkl','wb'))


# In[25]:


# Find scores that show how important each feature is

importance = model.feature_importances_

importances_dict = {}

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    importances_dict[i] = v

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[27]:


# Find how many are above >= 0.01

top = []

for i in importance:
    if i >= 0.01:
        top.append(i)

amount_top_features = len(top)
print(amount_top_features)


# In[30]:


# Sort features and extract top (scores >=  0.01)

sorted_features = sorted(importances_dict.items(), key = lambda x: x[1], reverse = True)

sorted_features_top = sorted_features[0:26]

top_features = []

for i in sorted_features_top:
    top_features.append(i[0])

# Find feature names

column_numbers = {}

for i,v in enumerate(data.columns):
    column_numbers[i] = v

# Match feature names and numbers
    
top_features_numbered = {}

for i in top_features:
    top_features_numbered[i] = column_numbers.get(i)
    

# Extract top feature names
    
top_important_features = []

for i in top_features_numbered.values():
    top_important_features.append(i)
    
top_important_features


# In[31]:


# Select only important features for a new training/test set

X_train = X_train[top_important_features]
X_test = X_test[top_important_features]


# In[32]:


# Retrain the model using important features

model = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = seed)

model.fit(X_train,y_train)


# In[33]:


# Make predictions on the test set

y_pred_test = model.predict(X_test)


# In[34]:


# Model evaluation / Confusion Matrix

confusion_m_test = confusion_matrix(y_test, y_pred_test)
confusion_m_test


# In[37]:


# Model evaluation and comparison / Accuracy, Precision, Recall, F1

accuracy_test = accuracy_score(y_test, y_pred_test)
print("Validation accuracy:", accuracy_val)
print("Test accuracy:", accuracy_test)

precision_test = precision_score(y_test, y_pred_test)
print("Validation precision:", precision_val)
print("Test precision:", precision_test)

recall_test = recall_score(y_test, y_pred_test)
print("Validation recall:", recall_val)
print("Test recall:", recall_test)

f1_test = f1_score(y_test, y_pred_test)
print("Validation f1:", f1_val)
print("Test f1:", f1_test)


# In[38]:


# Save this model

pickle.dump(model, open('top_features_model.pkl','wb'))


# In[39]:


# Prepare final test data

# Replace all missing values with 'na' which will be encoded

data_test = data_test.fillna('na')

# Create a no_hashtags column containing a number of hashtags for that listing 
# which will be used as a variable instead of the hashtags column (easier to measure)

def no_hashtags(hashtags):
    no_hashtags = int(hashtags.size)
    return no_hashtags

data_test['no_hashtags'] = data_test.apply(lambda row: no_hashtags(row['hashtags']), axis=1)


# In[40]:


# Select only the top important features for the test set

X_test_final = data_test[top_important_features]


# In[41]:


# Encode all object variables

for col_name in X_test_final.columns:
    if(X_test_final[col_name].dtype == 'object'):
        X_test_final[col_name]= X_test_final[col_name].astype('category')
        X_test_final[col_name] = X_test_final[col_name].cat.codes


# In[42]:


X_test_final.info()


# In[43]:


model = pickle.load(open('top_features_model.pkl', 'rb'))


# In[44]:


# Make predictions on the test set

y_pred_final = model.predict(X_test_final)


# In[45]:


# Add a predictions column to the test data then filter the rows that were predicted to be sold within 24 hours

X_test_final['predicted_label'] = y_pred_final


# In[46]:


X_final = X_test_final[X_test_final['predicted_label'] == 1]


# In[47]:


# 51 lines of the test dataset were predicted to be sold within 24 hours

X_final.shape


# In[48]:


# Drop the added column and save the dataframe

X_test_final = X_test_final.drop(['predicted_label'], axis = 1)

X_test_final.to_parquet('predictions.parquet')

