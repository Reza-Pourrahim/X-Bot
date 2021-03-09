#!/usr/bin/env python
# coding: utf-8

# In[57]:


import lore
from datamanager import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score

from util import record2str


# ipynb to py in pycharm:
# - jupyter nbconvert --to script new_run.ipynb

# ## Dataset

# In[66]:


## Iris Dataset
# dataset_name = 'dataset/iris.csv'
# dataset = prepare_iris_dataset(dataset_name)

## wine
# dataset_name = 'dataset/wine.csv'
# dataset = prepare_wine_dataset(dataset_name)

##############################################
#           Categorical dataset              #
##############################################
## german: (0 = Good, 1 = Bad)
# dataset_name = 'dataset/german_credit.csv'
# dataset = prepare_german_dataset(dataset_name)

## adult: ['<=50K', '>50K']
# dataset_name = 'dataset/adult.csv'
# dataset = prepare_adult_dataset(dataset_name)

## compas-scores-two-years: ['High', 'Low', 'Medium']
dataset_name = 'dataset/compas-scores-two-years.csv'
dataset = prepare_compass_dataset(dataset_name)

dataframe = dataset[0]
class_name = dataset[1]
dataset_fin = prepare_dataset(dataframe, class_name)


# In[67]:


df = dataset_fin[0] #dataframe with unique numeric class values(0, 1, ...)
feature_names = dataset_fin[1]
class_values = dataset_fin[2]
numeric_columns = dataset_fin[3]
rdf = dataset_fin[4] #real dataframe
real_feature_names = dataset_fin[5]
features_map = dataset_fin[6] #map each class name to its unique numeric value


# In[68]:


rdf.head()


# In[69]:


df.head()


# ## Black box classifier

# In[70]:


X = df.loc[:, df.columns != class_name].values
y = df[class_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
blackbox = RandomForestClassifier()
blackbox.fit(X_train, y_train)


# In[71]:


y_pred = blackbox.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# ## select an instance _x_

# In[72]:


i = 10
x = X_test[i]
y_val = blackbox.predict(x.reshape(1,-1))[0]

print(class_values)
class_prob = blackbox.predict_proba(x.reshape(1,-1))[0]
print(class_prob)

y_val_name = class_values[y_val]
print('blackbox(x) = { %s }' % y_val_name)


# In[73]:


print('x = %s' % record2str(x, feature_names, numeric_columns))


# # LORE explainer (explaining an instance x)

# In[78]:


lore_obj = lore.LORE(X_test, blackbox, feature_names, class_name, class_values,
                 numeric_columns, features_map, neigh_type='ngmusx', verbose=False)


# In[79]:


# just to check
Z = lore_obj.neighgen_fn(x)
print('Z is:',Z)
Z.shape


# In[80]:


explanation = lore_obj.explain_instance(x, samples=1000, nbr_runs=10)

print(explanation)


# In[81]:


explanation = lore_obj.explain_instance(x, samples=1000, nbr_runs=10)

print(explanation)


# In[82]:


print('x = %s' % record2str(x, feature_names, numeric_columns))



