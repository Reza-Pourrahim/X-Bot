import pickle

from lorem import LOREM
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
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


# BlackBox TO EXPLAIN
# blackbox = RandomForestClassifier()
# blackbox = SGDClassifier(loss='log')
# blackbox = SVC(probability=True)
blackbox = GradientBoostingClassifier()


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
# dataset_name = 'explainer_dataset/wine.csv'
# explainer_dataset = prepare_wine_dataset(dataset_name)
# pkl_explainer_object = 'blackbox_model_files/wine_explainer_object.pkl'

# pkl_filename = 'blackbox_model_files/wine_RandomForestClassifier.pkl'
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


## adult: ['<=50K', '>50K'] -> [0, 1]
# dataset_name = 'explainer_dataset/adult.csv'
# explainer_dataset = prepare_adult_dataset(dataset_name)
# pkl_explainer_object = 'blackbox_model_files/adult_explainer_object.pkl'
#
# pkl_filename = 'blackbox_model_files/adult_RandomForestClassifier.pkl'
# pkl_filename = 'blackbox_model_files/adult_SGDClassifier.pkl'
# pkl_filename = 'blackbox_model_files/adult_SVC.pkl'
# pkl_filename = 'blackbox_model_files/adult_GradientBoostingClassifier.pkl'

## compas-scores-two-years: ['High', 'Low', 'Medium'] -> [0, 1, 2]
dataset_name = 'explainer_dataset/compas-scores-two-years.csv'
explainer_dataset = prepare_compass_dataset(dataset_name)
pkl_explainer_object = 'blackbox_model_files/compas_explainer_object.pkl'

# pkl_filename = 'blackbox_model_files/compas_RandomForestClassifier.pkl'
# pkl_filename = 'blackbox_model_files/compas_SGDClassifier.pkl'
# pkl_filename = 'blackbox_model_files/compas_SVC.pkl'
pkl_filename = 'blackbox_model_files/compas_GradientBoostingClassifier.pkl'


dataframe, class_name = explainer_dataset
if dataset_name == 'explainer_dataset/iris.csv':
    new_values = list()
    for f in dataframe[class_name]:
        if f == 'Iris-setosa':
            new_values.append(0)
        elif f == "Iris-versicolor":
            new_values.append(1)
        else:
            new_values.append(2)
    dataframe[class_name] = new_values

if dataset_name == 'explainer_dataset/adult.csv':
    new_values = list()
    for f in dataframe[class_name]:
        if f == '<=50K':
            new_values.append(0)
        else:
            new_values.append(1)
    dataframe[class_name] = new_values

if dataset_name == 'explainer_dataset/compas-scores-two-years.csv':
    new_values = list()
    for f in dataframe[class_name]:
        if f == 'High':
            new_values.append(0)
        elif f == "Low":
            new_values.append(1)
        else:
            new_values.append(2)
    dataframe[class_name] = new_values

# features = [c for c in dataframe.columns if c not in [class_name]]
# cont_features_names = list(dataframe[features]._get_numeric_data().columns)
# cate_features_names = [c for c in dataframe.columns if c not in cont_features_names and c != class_name]
# y = dataframe[[class_name]]
# df1 = dataframe[features]
# cont_features_idx = [features.index(f) for f in cont_features_names]
# cate_features_idx = [features.index(f) for f in cate_features_names]
# ohe = OneHotEncoder()
# X_cat = ohe.fit_transform(df1.iloc[:, cate_features_idx].values)

# X_cat = pd.get_dummies(df1[[c for c in df1.columns if c != class_name]], prefix_sep='=')
# cont_values = df1.iloc[:, cont_features_idx]
# X is now the dataset encoded to use for the train of the classifier
# X = pd.concat([cont_values, X_cat], axis=1)


dataset_fin = prepare_dataset(dataframe, class_name, encdec='onehot')
df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map, df_categorical_idx,\
df_numeric_idx = dataset_fin
X_new = df.loc[:, df.columns != class_name].values
y_new = df[class_name].values


X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.30, random_state=42)
# Training
blackbox.fit(X_train, y_train)


# save the model to disk
with open(pkl_filename, 'wb') as file:
    pickle.dump(blackbox, file)



#################
#   SuperLORE   #
#################
# dataset_fin = prepare_dataset(dataframe, class_name, encdec='target')
#
# df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map, df_categorical_idx,\
# df_numeric_idx = dataset_fin
#
# part_data = df.iloc[X_test.index, :]
# part_data[class_name] = y_test
# nuovo_data = df.iloc[X_test.index, :]
# nuovo_data[class_name] = y_test
# part_data.pop(class_name)

# neigh_kwargs = {
#     "balance": False,
#     "sampling_kind": "uniform_sphere",
#     "kind": "uniform_sphere",
#     "downward_only": True,
#     "redo_search": True,
#     "forced_balance_ratio": 0.5,
#     "cut_radius": True,
#     "verbose": True,
#     "n":6000,
#     "n_batch": 50000,
#     "normalize" : 'minmax',
#     "datas" : nuovo_data
# }
# threshold_kwargs = {
#     "upper_threshold" :4,
#     "lower_threshold" :0,
#     "n_search" : 300000,
#     "stopping_ratio": 0.5
# }
# def bb_predict(X):
#     X_cat = ohe.transform(X[:, cate_features_idx])
#     X_cont = X[:, cont_features_idx]
#     X = np.concatenate((X_cont, X_cat), axis=1)
#     if len(X.shape) == 1:
#         X = X.reshape(1, -1)
#     return blackbox.predict(X)
#
# def bb_predict_proba(X):
#     X_cat = ohe.transform(X[:, cate_features_idx])
#     X_cont = X[:, cont_features_idx]
#     X = np.concatenate((X_cont, X_cat), axis=1)
#     if len(X.shape) == 1:
#         X = X.reshape(1, -1)
#     return blackbox.predict_proba(X)

# explainer object
# explainer_object = LOREM(part_data.values, bb_predict, feature_names, class_name,
#                        class_values, numeric_columns, features_map,
#                        neigh_type='cfs', categorical_use_prob=True,
#                        continuous_fun_estimation=False,
#                        size=3000, ocr=0.1, random_state=42,
#                        bb_predict_proba=bb_predict_proba,
#                        filter_crules=False,
#                        binary=True,
#                        encdec='target',
#                        dataset=nuovo_data,
#                        discretize=False,
#                        **neigh_kwargs)
#
# with open(pkl_explainer_object, 'wb') as f:
#     pickle.dump(explainer_object, f, pickle.HIGHEST_PROTOCOL)
