from xbot_model import XBotModel
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import StratifiedKFold

kf = KFold(n_splits=10)
skf = StratifiedKFold(n_splits=10)

# LORE X-BOT
data_file = open('chat_dataset/intents.json').read()
data = json.loads(data_file)

XBot_obj = XBotModel(verbose=True)
train_dataset, vocab_size, tokenizer, embedding_matrix = XBot_obj.prepare_train_dataset(data)
train_x = list(train_dataset[:, 0])
train_y = list(train_dataset[:, 1])

def mergeDict(dict1, dict2):
   ''' Merge dictionaries and keep values of common keys in list'''
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = [value , dict1[key]]
   return dict3

with open('chatbot_model_files/classes.pkl', 'rb') as f:
    target_names = pickle.load(f)

cvscores = []
f1=[]
recall=[]
prec=[]


train_y_df = pd.DataFrame(np.array(train_y))
train_y_lst = train_y_df.idxmax(axis=1).values.tolist()
# X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)
for train_index, test_index, in skf.split(train_x, train_y_lst):

    X_train , X_test = [train_x[i] for i in train_index], [train_x[i] for i in test_index]
    y_train , y_test = [train_y[i] for i in train_index], [train_y[i] for i in test_index]
    y_test_df = pd.DataFrame(np.array(y_test))
    y_test_lst = y_test_df.idxmax(axis=1).values.tolist()
    # if len(np.unique(y_test_lst)) != len(target_names):
    #     continue
    # counter = counter+1
    # if counter > 10:
    #     break
    model = XBot_obj.create_model(X_train, y_train, embedding_matrix, vocab_size,
                                embedding_dim=100, lstm_out=15, dropout=0.5)
    mymodel = XBot_obj.compile_fit_model(model, X_train, y_train, epochs=100,
                                       batch_size=5,
                                       earlystopping_patience=10,
                                       validation_split=0.2,
                                       loss='categorical_crossentropy')

    trained_model = load_model('chatbot_model_files/best_xbot_model.h5')
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    scores = trained_model.evaluate(np.array(X_test), np.array(y_test), batch_size=5)
    # scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    y_pred = trained_model.predict(np.array(X_test), batch_size=5, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    # y_test_df = pd.DataFrame(np.array(y_test))
    # y_test_lst = y_test_df.idxmax(axis=1).values.tolist()
    f1.append(f1_score(y_test_lst, y_pred_bool, average='weighted'))
    recall.append(recall_score(y_test_lst, y_pred_bool, average='weighted'))
    prec.append(precision_score(y_test_lst, y_pred_bool, average='weighted'))
    print('f1: ', f1)
    print('recall: ', recall)
    print('prec: ', prec)
    # dict = classification_report(y_test_lst, y_pred_bool, target_names=target_names, output_dict=True)
    # table_related = mergeDict(table_related, dict)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(prec), np.std(prec)))
pickle.dump(f1, open('chatbot_model_files/f1.pkl', 'wb'))
pickle.dump(recall, open('chatbot_model_files/recall.pkl', 'wb'))
pickle.dump(prec, open('chatbot_model_files/prec.pkl', 'wb'))

#
# bot_challenge = {}
# counter_exemplar = {}
# exemplar = {}
# feature_importance = {}
# goodbye = {}
# greet = {}
# how_to_be_that = {}
# options = {}
# performance = {}
# why = {}
#
# for i in range(10):
#     bot_challenge = mergeDict(table_related.get('bot_challenge')[i], bot_challenge)
#     counter_exemplar = mergeDict(table_related.get('counter_exemplar')[i], counter_exemplar)
#     exemplar = mergeDict(table_related.get('exemplar')[i], exemplar)
#     feature_importance = mergeDict(table_related.get('feature_importance')[i], feature_importance)
#     goodbye = mergeDict(table_related.get('goodbye')[i], goodbye)
#     greet = mergeDict(table_related.get('greet')[i], greet)
#     how_to_be_that = mergeDict(table_related.get('how_to_be_that')[i], how_to_be_that)
#     options = mergeDict(table_related.get('options')[i], options)
#     performance = mergeDict(table_related.get('performance')[i], performance)
#     why = mergeDict(table_related.get('why')[i], why)
#
# for item in why.keys():
#     for item2 in why.values():
#         print(item2)
