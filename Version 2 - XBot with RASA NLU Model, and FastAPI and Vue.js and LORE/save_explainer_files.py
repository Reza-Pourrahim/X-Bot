import pickle

import lore
from datamanager import *

# Random Forest
from sklearn.ensemble import RandomForestClassifier
#Support Vector Machine
from sklearn.svm import SVC
#Naive Bayes (Gaussian)
from sklearn.naive_bayes import GaussianNB
#Stochastic Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier
#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

# BlackBox TO EXPLAIN
# blackbox = GradientBoostingClassifier()
blackbox = RandomForestClassifier()
# blackbox = SGDClassifier(loss='log')
# blackbox = SVC(probability=True)


# DATASET TO EXPLAIN
# Iris Dataset
# dataset_name = 'explainer_dataset/iris.csv'
# explainer_dataset = prepare_iris_dataset(dataset_name)
# pkl_explainer_object = 'blackbox_model_files/iris_explainer_object.pkl'

# pkl_filename = 'blackbox_model_files/iris_RandomForestClassifier.pkl'
# pkl_filename = 'blackbox_model_files/iris_SGDClassifier.pkl'
# pkl_filename = 'blackbox_model_files/iris_SVC.pkl'
# pkl_filename = 'blackbox_model_files/iris_GradientBoostingClassifier.pkl'


## wine
dataset_name = 'explainer_dataset/wine.csv'
explainer_dataset = prepare_wine_dataset(dataset_name)
pkl_explainer_object = 'blackbox_model_files/wine_explainer_object.pkl'

pkl_filename = 'blackbox_model_files/wine_RandomForestClassifier.pkl'
# pkl_filename = 'blackbox_model_files/wine_SGDClassifier.pkl'
# pkl_filename = 'blackbox_model_files/wine_SVC.pkl'
# pkl_filename = 'blackbox_model_files/wine_GradientBoostingClassifier.pkl'


##############################################
#           Categorical explainer_dataset     #
##############################################
## german: (0 = Good, 1 = Bad)
# dataset_name = 'explainer_dataset/german_credit.csv'
# explainer_dataset = prepare_german_dataset(dataset_name)
# pkl_explainer_object = 'blackbox_model_files/german_credit_explainer_object.pkl'

# pkl_filename = 'blackbox_model_files/german_credit_RandomForestClassifier.pkl'
# pkl_filename = 'blackbox_model_files/german_credit_SGDClassifier.pkl'
# pkl_filename = 'blackbox_model_files/german_credit_SVC.pkl'
# pkl_filename = 'blackbox_model_files/german_credit_GradientBoostingClassifier.pkl'


## adult: ['<=50K', '>50K']
# dataset_name = 'explainer_dataset/adult.csv'
# explainer_dataset = prepare_adult_dataset(dataset_name)
# pkl_explainer_object = 'blackbox_model_files/adult_explainer_object.pkl'

# pkl_filename = 'blackbox_model_files/adult_RandomForestClassifier.pkl'
# pkl_filename = 'blackbox_model_files/adult_SGDClassifier.pkl'
# pkl_filename = 'blackbox_model_files/adult_SVC.pkl'
# pkl_filename = 'blackbox_model_files/adult_GradientBoostingClassifier.pkl'


## compas-scores-two-years: ['High', 'Low', 'Medium']
# dataset_name = 'explainer_dataset/compas-scores-two-years.csv'
# explainer_dataset = prepare_compass_dataset(dataset_name)
# pkl_explainer_object = 'blackbox_model_files/compas_explainer_object.pkl'

# pkl_filename = 'blackbox_model_files/compas_RandomForestClassifier.pkl'
# pkl_filename = 'blackbox_model_files/compas_SGDClassifier.pkl'
# pkl_filename = 'blackbox_model_files/compas_SVC.pkl'
# pkl_filename = 'blackbox_model_files/compas_GradientBoostingClassifier.pkl'


dataframe, class_name = explainer_dataset
dataset_fin = prepare_dataset(dataframe, class_name)

df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map, df_categorical_idx \
    = dataset_fin

X = df.loc[:, df.columns != class_name].values
y = df[class_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
# Training
blackbox.fit(X_train, y_train)


# save the model to disk
with open(pkl_filename, 'wb') as file:
    pickle.dump(blackbox, file)

# explainer object
explainer_object = lore.LORE(X_test, blackbox, feature_names, class_name, class_values, numeric_columns, features_map,
                             df_categorical_idx, neigh_type='ngmusx', verbose=False)
with open(pkl_explainer_object, 'wb') as f:
    pickle.dump(explainer_object, f, pickle.HIGHEST_PROTOCOL)
