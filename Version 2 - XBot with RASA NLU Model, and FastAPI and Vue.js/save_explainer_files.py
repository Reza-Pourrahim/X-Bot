import pickle

import lore
from datamanager import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# DATASET FOR EXPLAIN
# Iris Dataset
# dataset_name = 'explainer_dataset/iris.csv'
# explainer_dataset = prepare_iris_dataset(dataset_name)
# pkl_filename = 'blackbox_model_files/iris_model.pkl'
# pkl_explainer_object = 'blackbox_model_files/iris_explainer_object.pkl'

## wine
# dataset_name = 'explainer_dataset/wine.csv'
# explainer_dataset = prepare_wine_dataset(dataset_name)
# pkl_filename = 'blackbox_model_files/wine_model.pkl'
# pkl_explainer_object = 'blackbox_model_files/wine_explainer_object.pkl'

##############################################
#           Categorical explainer_dataset              #
##############################################
## german: (0 = Good, 1 = Bad)
dataset_name = 'explainer_dataset/german_credit.csv'
explainer_dataset = prepare_german_dataset(dataset_name)
pkl_filename = 'blackbox_model_files/german_credit_model.pkl'
pkl_explainer_object = 'blackbox_model_files/german_credit_explainer_object.pkl'


## adult: ['<=50K', '>50K']
# dataset_name = 'explainer_dataset/adult.csv'
# explainer_dataset = prepare_adult_dataset(dataset_name)
# pkl_filename = 'blackbox_model_files/adult_model.pkl'
# pkl_explainer_object = 'blackbox_model_files/adult_explainer_object.pkl'


## compas-scores-two-years: ['High', 'Low', 'Medium']
# dataset_name = 'explainer_dataset/compas-scores-two-years.csv'
# explainer_dataset = prepare_compass_dataset(dataset_name)
# pkl_filename = 'blackbox_model_files/compas_model.pkl'
# pkl_explainer_object = 'blackbox_model_files/compas_explainer_object.pkl'


dataframe, class_name = explainer_dataset
dataset_fin = prepare_dataset(dataframe, class_name)

df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map, df_categorical_idx \
    = dataset_fin

X = df.loc[:, df.columns != class_name].values
y = df[class_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
blackbox = RandomForestClassifier()
blackbox.fit(X_train, y_train)

# save the model to disk
with open(pkl_filename, 'wb') as file:
    pickle.dump(blackbox, file)

# explainer object
explainer_object = lore.LORE(X_test, blackbox, feature_names, class_name, class_values, numeric_columns, features_map,
                             df_categorical_idx, neigh_type='ngmusx', verbose=False)
with open(pkl_explainer_object, 'wb') as f:
    pickle.dump(explainer_object, f, pickle.HIGHEST_PROTOCOL)
