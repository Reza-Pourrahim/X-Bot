#!/usr/bin/env python
# coding: utf-8

# In[43]:


import lore
from datamanager import *

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


# ### ipynb to py in pycharm:
# jupyter nbconvert --to script new_run.ipynb

# In[44]:


## Iris Dataset
dataset_name = 'dataset/iris.csv'
dataset = prepare_iris_dataset(dataset_name)

## german
# dataset_name = 'dataset/german_credit.csv'
# dataset = prepare_german_dataset(dataset_name)

## adult
# dataset_name = 'dataset/adult.csv'
# dataset = prepare_adult_dataset(dataset_name)

## compas-scores-two-years
# dataset_name = 'dataset/compas-scores-two-years.csv'
# dataset = prepare_compass_dataset(dataset_name)

## wine
# dataset_name = 'dataset/wine.csv'
# dataset = prepare_wine_dataset(dataset_name)

dataframe = dataset[0]
class_name = dataset[1]
dataset_fin = prepare_dataset(dataframe, class_name)


# In[45]:


dataframe.head()


# In[46]:


df = dataset_fin[0] #dataframe with unique numeric class values(0, 1, ...)
feature_names = dataset_fin[1]
class_values = dataset_fin[2]
numeric_columns = dataset_fin[3]
rdf = dataset_fin[4] #real dataframe
real_feature_names = dataset_fin[5]
features_map = dataset_fin[6] #map each class name to its unique numeric value

X = df.loc[:, df.columns != class_name].values
y = df[class_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
blackbox = RandomForestClassifier()
blackbox.fit(X_train, y_train)


# In[47]:


y_pred = blackbox.predict(X_test)
print('Accuracy %.3f' % accuracy_score(y_test, y_pred))


# In[48]:


i = 10
x = X_test[i]
y_val = blackbox.predict(x.reshape(1,-1))[0]

y_val_name = class_values[y_val]
print(y_val_name)


# In[49]:


lore_obj = lore.LORE(X_test, blackbox, feature_names, class_name, class_values,
                 numeric_columns, features_map, neigh_type='gmusx', verbose=True)

Z = lore_obj.neighgen_fn(x)

print('Z is:', Z)


# In[50]:


# explanation = lore_obj.explain_instance(x, y_val, blackbox, nbr_runs=10, verbose=True)
#
# print(explanation)




