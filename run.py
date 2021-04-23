from xbot_model import XBotModel
from response_generator import ResponseGenerator

from keras.models import load_model

import json
import matplotlib.pyplot as plt
import lore
from datamanager import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from util import record2str


# ipynb to py in pycharm:
# - jupyter nbconvert --to script new_run_with_keras.ipynb

# ## Dataset

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


df = dataset_fin[0] #dataframe with unique numeric class values(0, 1, ...)
feature_names = dataset_fin[1]
class_values = dataset_fin[2]
numeric_columns = dataset_fin[3]
rdf = dataset_fin[4] #real dataframe
real_feature_names = dataset_fin[5]
features_map = dataset_fin[6] #map each class name to its unique numeric value
df_categorical_idx = dataset_fin[7]


# ## Black box classifier

X = df.loc[:, df.columns != class_name].values
y = df[class_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
blackbox = RandomForestClassifier()
blackbox.fit(X_train, y_train)


y_pred = blackbox.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# ## select an instance _x_

# i = 10
# x = X_test[i]
# y_val = blackbox.predict(x.reshape(1,-1))[0]
#
# print(class_values)
# class_prob = blackbox.predict_proba(x.reshape(1,-1))[0]
# print(class_prob)
#
# y_val_name = class_values[y_val]
# print('blackbox(x) = { %s }' % y_val_name)
#
#
# print('x = %s' % record2str(x, feature_names, numeric_columns))


# # LORE explainer (explaining an instance x)
explainer_obj = lore.LORE(X_test, blackbox, feature_names, class_name, class_values,
                 numeric_columns, features_map, df_categorical_idx, neigh_type='ngmusx', verbose=False)



# just to check
# Z = lore_obj.neighgen_fn(x, categorical_columns=df_categorical_idx)
# print('Z is:',Z)
# Z.shape


# explanation = explainer_obj.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)
# print(explanation)



# # X-Bot

###################### Chat Data ######################
data_file = open('chatdata/intents.json').read()
data = json.loads(data_file)


# ## MODEL
XBot_obj = XBotModel(verbose=True)

# training dataset
train_dataset, vocab_size, tokenizer, embedding_matrix = XBot_obj.prepare_train_dataset(data)
train_x = list(train_dataset[:, 0])
train_y = list(train_dataset[:, 1])


# create model
model = XBot_obj.create_model(train_x, train_y, embedding_matrix, vocab_size,
                              embedding_dim=100, lstm_out=15, dropout=0.5)


# compile and fit the model
mymodel = XBot_obj.compile_fit_model(model, train_x, train_y, epochs=200,
                                     batch_size=5,
                                     earlystopping_patience=10,
                                     validation_split=0.3,
                                     loss='categorical_crossentropy')

# plot the Train and Validation loss
plt.plot(mymodel['loss'], label='Train')
plt.plot(mymodel['val_loss'], label='Val')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy')
plt.legend()
plt.show()

# plot the Train and Validation accuracy
plt.plot(mymodel['accuracy'], label='Train')
plt.plot(mymodel['val_accuracy'], label='Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


trained_model = load_model('best_xbot_model.h5')

_, accuracy = trained_model.evaluate(np.array(train_x), np.array(train_y))
print('Train Accuracy: %.2f' % (accuracy*100))

xbot_response = ResponseGenerator(data, train_x, train_y, tokenizer, trained_model,
                                  X, dataset_fin, explainer_obj, blackbox, verbose=True)

xbot_response.start()