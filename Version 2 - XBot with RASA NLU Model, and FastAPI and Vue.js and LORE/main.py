from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]

import os
import random
import json
from explanation import ExplanationEncoder

import numpy as np
import pickle

# Version 1 - Bidirectional LSTM and Embedding Model
# import tensorflow as tf
# from response_generator import ResponseGenerator

# Version 2 - RASA Model
from rasa.nlu.model import Interpreter


###################
# files and lists #
###################
##########
# Adult #
##########
pkl_explainer_object_adult = 'blackbox_model_files/adult_explainer_object.pkl'
adult_class_values = ['<=50K', '>50K']
# Load the Model back from file - adult
with open(pkl_explainer_object_adult, 'rb') as f:
    explainer_object_adult = pickle.load(f)

##########
# COMPAS #
##########
pkl_explainer_object_compas = 'blackbox_model_files/compas_explainer_object.pkl'
compas_class_values = ['High', 'Low', 'Medium']
# Load the Model back from file - compas
with open(pkl_explainer_object_compas, 'rb') as f:
    explainer_object_compas = pickle.load(f)

########
# Iris #
########
pkl_explainer_object_iris = 'blackbox_model_files/iris_explainer_object.pkl'
iris_class_values = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# Load the Model back from file - iris
with open(pkl_explainer_object_iris, 'rb') as f:
    explainer_object_iris = pickle.load(f)

#################
# German Credit #
#################
pkl_explainer_object_german = 'blackbox_model_files/german_credit_explainer_object.pkl'
# (0 = Good, 1 = Bad)
german_class_values = ['Good', 'Bad']
# Load the Model back from file - iris
with open(pkl_explainer_object_german, 'rb') as f:
    explainer_object_german = pickle.load(f)

########
# Wine #
########
pkl_explainer_object_wine = 'blackbox_model_files/wine_explainer_object.pkl'
wine_class_values = [1, 2, 3]
# Load the Model back from file - wine
with open(pkl_explainer_object_wine, 'rb') as f:
    explainer_object_wine = pickle.load(f)


#########################
# Intents and Responses #
#########################
explanation_intents = ['why', 'performance','how_to_be_that','feature_importance','exemplar','counter_exemplar']
greet_response = ["Ciao", "Buongiorno", "Hi", "Hello", "Good to see you again", "Hi there, how can I help?"]
goodbye_response = ["See you!", "Have a nice day", "Bye! Come back again soon."]
bot_challenge_response = ["Sorry, can't understand you", "Please give me more information", "Not sure I understand"]
options_response = ["I can give you an explanation consisting of a decision rule, explaining the factual reasons of the decision, and a set of counterfactuals, suggesting the changes in the instance features that would lead to a different outcome! Furthermore, I can Say you the most important feature, bring to you some exemplars, and some counter exemplar to clarify it more!"]



###########
# FASTAPI #
###########
# pip install fastapi
# pip install uvicorn[standard]
# uvicorn main:app --reload
# http://127.0.0.1:8000/docs
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"]
)


from fastapi.responses import FileResponse

# @app.get("/app")
# def read_index():
#     return FileResponse("./app.html")

######################################################
# Version 1 - Bidirectional LSTM and Embedding Model #
######################################################
# Load Chat_bot Files
# data_file = open('chat_dataset/intents.json').read()
# data = json.loads(data_file)
#
# xbot_trained_model = tf.keras.models.load_model('chatbot_model_files/best_xbot_model.h5')
# with open('chatbot_model_files/tokenizer.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)
# embedding_matrix = np.load('chatbot_model_files/embedding_matrix.npy')
# training_dataset = np.load('chatbot_model_files/training_dataset.npy', allow_pickle=True)
#
# train_x = list(training_dataset[:, 0])
# train_y = list(training_dataset[:, 1])

# xbot_response = ResponseGenerator(data, train_x, train_y, tokenizer, xbot_trained_model,
#                                   explainer_object, verbose=False)


##########################
# Version 2 - RASA Model #
##########################
# pip3 install rasa[full]
# rasa init
# rasa train
# rasa shell
interpreter = Interpreter.load(
    os.path.join('models', 'nlu'),  # Had to extract the tar.gz model file before running this code
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def write_explanation_to_file(explanation):
    # if needed #
    # with open('blackbox_model_files/explanation.json', 'w') as jsonfile:
    #     json.dump(explanation, jsonfile, cls=ExplanationEncoder)

    verbatim_explanation = str(explanation)
    f = open("blackbox_model_files/verbatim_explanation.txt", "w")
    f.write(verbatim_explanation)
    f.close()


def predict_class(x, blackbox, class_values):
    y_val = blackbox.predict(x.reshape(1, -1))[0]
    res = class_values[y_val]
    return res


def load_blackbox_adult(model_to_explain):
    if model_to_explain == 'RandomForestClassifier':
        file_name = 'blackbox_model_files/adult_RandomForestClassifier.pkl'
    elif model_to_explain == 'SGDClassifier':
        file_name = 'blackbox_model_files/adult_SGDClassifier.pkl'
    elif model_to_explain == 'SVC':
        file_name = 'blackbox_model_files/adult_SVC.pkl'
    elif model_to_explain == 'GradientBoostingClassifier':
        file_name = 'blackbox_model_files/adult_GradientBoostingClassifier.pkl'
    with open(file_name, 'rb') as file:
        blackbox = pickle.load(file)
    return blackbox


def load_blackbox_compas(model_to_explain):
    if model_to_explain == 'RandomForestClassifier':
        file_name = 'blackbox_model_files/compas_RandomForestClassifier.pkl'
    elif model_to_explain == 'SGDClassifier':
        file_name = 'blackbox_model_files/compas_SGDClassifier.pkl'
    elif model_to_explain == 'SVC':
        file_name = 'blackbox_model_files/compas_SVC.pkl'
    elif model_to_explain == 'GradientBoostingClassifier':
        file_name = 'blackbox_model_files/compas_GradientBoostingClassifier.pkl'
    with open(file_name, 'rb') as file:
        blackbox = pickle.load(file)
    return blackbox



def load_blackbox_german_credit(model_to_explain):
    if model_to_explain == 'RandomForestClassifier':
        file_name = 'blackbox_model_files/german_credit_RandomForestClassifier.pkl'
    elif model_to_explain == 'SGDClassifier':
        file_name = 'blackbox_model_files/german_credit_SGDClassifier.pkl'
    elif model_to_explain == 'SVC':
        file_name = 'blackbox_model_files/german_credit_SVC.pkl'
    elif model_to_explain == 'GradientBoostingClassifier':
        file_name = 'blackbox_model_files/german_credit_GradientBoostingClassifier.pkl'
    with open(file_name, 'rb') as file:
        blackbox = pickle.load(file)
    return blackbox


def load_blackbox_iris(model_to_explain):
    if model_to_explain == 'RandomForestClassifier':
        file_name = 'blackbox_model_files/iris_RandomForestClassifier.pkl'
    elif model_to_explain == 'SGDClassifier':
        file_name = 'blackbox_model_files/iris_SGDClassifier.pkl'
    elif model_to_explain == 'SVC':
        file_name = 'blackbox_model_files/iris_SVC.pkl'
    elif model_to_explain == 'GradientBoostingClassifier':
        file_name = 'blackbox_model_files/iris_GradientBoostingClassifier.pkl'
    with open(file_name, 'rb') as file:
        blackbox = pickle.load(file)
    return blackbox


def load_blackbox_wine(model_to_explain):
    if model_to_explain == 'RandomForestClassifier':
        file_name = 'blackbox_model_files/wine_RandomForestClassifier.pkl'
    elif model_to_explain == 'SGDClassifier':
        file_name = 'blackbox_model_files/wine_SGDClassifier.pkl'
    elif model_to_explain == 'SVC':
        file_name = 'blackbox_model_files/wine_SVC.pkl'
    elif model_to_explain == 'GradientBoostingClassifier':
        file_name = 'blackbox_model_files/wine_GradientBoostingClassifier.pkl'
    with open(file_name, 'rb') as file:
        blackbox = pickle.load(file)
    return blackbox


@app.get("/iris_lore")
def explain_iris(model_to_explain: str = 'RandomForestClassifier', sepal_length: float = 0, sepal_width: float = 0,
                 petal_length: float = 0, petal_width: float = 0):
    x = [sepal_length, sepal_width, petal_length, petal_width]
    x = np.array(x)

    blackbox = load_blackbox_iris(model_to_explain)

    class_iris = predict_class(x, blackbox, iris_class_values)
    class_prob = max(blackbox.predict_proba(x.reshape(1, -1))[0])
    # numpy_file = "blackbox_model_files/instance_x.npy"
    # np.save(numpy_file, x)

    # explanation = explainer_object_iris.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)
    # write_explanation_to_file(explanation)

    return {'class_iris': class_iris,
            'class_prob': class_prob*100
            }


@app.get("/iris_lore_explanation")
def iris_lore_explanation(model_to_explain: str = 'RandomForestClassifier', sepal_length: float = 0, sepal_width: float = 0,
                 petal_length: float = 0, petal_width: float = 0):
    x = [sepal_length, sepal_width, petal_length, petal_width]
    x = np.array(x)
    # numpy_file = "blackbox_model_files/instance_x.npy"
    # x = np.load(numpy_file)

    explanation = explainer_object_iris.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)

    write_explanation_to_file(explanation)

    return True


@app.get("/german_lore")
def explain_german(model_to_explain: str = 'RandomForestClassifier', duration_in_month: int = 0, credit_amount: int = 0, installment_as_income_perc: int = 0,
                   present_res_since: int = 0, age: int = 0, credits_this_bank: int = 0,
                   people_under_maintenance: int = 0, account_check_status: str = '0 <= ... < 200 DM',
                   credit_history: str = 'all credits at this bank paid back duly',
                   purpose: str = '(vacation - does not exist?)', savings: str = '.. >= 1000 DM ',
                   present_emp_since: str = '.. >= 7 years',
                   personal_status_sex: str = 'female : divorced/separated/married', other_debtors: str = 'co-applicant',
                   property: str = 'if not A121 : building society savings agreement/ life insurance',
                   other_installment_plans: str = 'bank', housing: str = 'for free',
                   job: str = 'management/ self-employed/ highly qualified employee/ officer', telephone: str = 'none',
                   foreign_worker: str = 'no'):
    x = [duration_in_month, credit_amount, installment_as_income_perc, present_res_since, age, credits_this_bank,
         people_under_maintenance]

    if account_check_status == '0 <= ... < 200 DM':
        x.extend([1, 0, 0, 0])
    elif account_check_status == '< 0 DM':
        x.extend([0, 1, 0, 0])
    elif account_check_status == '>= 200 DM / salary assignments for at least 1 year':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])


    if credit_history == 'all credits at this bank paid back duly':
        x.extend([1, 0, 0, 0, 0])
    elif credit_history == 'critical account/ other credits existing (not at this bank)':
        x.extend([0, 1, 0, 0, 0])
    elif credit_history == 'delay in paying off in the past':
        x.extend([0, 0, 1, 0, 0])
    elif credit_history == 'existing credits paid back duly till now':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])


    if purpose == '(vacation - does not exist?)':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif purpose == 'business':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif purpose == 'car (new)':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif purpose == 'car (used)':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif purpose == 'domestic appliances':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif purpose == 'education':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif purpose == 'furniture/equipment':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif purpose == 'radio/television':
        x.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif purpose == 'repairs':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


    if savings == '.. >= 1000 DM ':
        x.extend([1, 0, 0, 0, 0])
    elif savings == '... < 100 DM':
        x.extend([0, 1, 0, 0, 0])
    elif savings == '100 <= ... < 500 DM':
        x.extend([0, 0, 1, 0, 0])
    elif savings == '500 <= ... < 1000 DM ':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])


    if present_emp_since == '.. >= 7 years':
        x.extend([1, 0, 0, 0, 0])
    elif present_emp_since == '... < 1 year ':
        x.extend([0, 1, 0, 0, 0])
    elif present_emp_since == '1 <= ... < 4 years':
        x.extend([0, 0, 1, 0, 0])
    elif present_emp_since == '4 <= ... < 7 years':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])


    if personal_status_sex == 'female : divorced/separated/married':
        x.extend([1, 0, 0, 0])
    elif personal_status_sex == 'male : divorced/separated':
        x.extend([0, 1, 0, 0])
    elif personal_status_sex == 'male : married/widowed':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])


    if other_debtors == 'co-applicant':
        x.extend([1, 0, 0])
    elif other_debtors == 'guarantor':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])


    if property == 'if not A121 : building society savings agreement/ life insurance':
        x.extend([1, 0, 0, 0])
    elif property == 'if not A121/A122 : car or other, not in attribute 6':
        x.extend([0, 1, 0, 0])
    elif property == 'real estate':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])


    if other_installment_plans == 'bank':
        x.extend([1, 0, 0])
    elif other_installment_plans == 'none':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])


    if housing == 'for free':
        x.extend([1, 0, 0])
    elif housing == 'own':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])


    if job == 'management/ self-employed/ highly qualified employee/ officer':
        x.extend([1, 0, 0, 0])
    elif job == 'skilled employee / official':
        x.extend([0, 1, 0, 0])
    elif job == 'unemployed/ unskilled - non-resident':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])


    if telephone == 'none':
        x.extend([1, 0])
    else:
        x.extend([0, 1])


    if foreign_worker == 'no':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    x = np.array(x)

    blackbox = load_blackbox_german_credit(model_to_explain)

    class_german = predict_class(x, blackbox, german_class_values)
    class_prob = max(blackbox.predict_proba(x.reshape(1, -1))[0])

    return {'class_german': class_german,
            'class_prob': class_prob * 100,
            }

@app.get("/german_lore_explanation")
def german_lore_explanation(model_to_explain: str = 'RandomForestClassifier', duration_in_month: int = 0, credit_amount: int = 0, installment_as_income_perc: int = 0,
                   present_res_since: int = 0, age: int = 0, credits_this_bank: int = 0,
                   people_under_maintenance: int = 0, account_check_status: str = '0 <= ... < 200 DM',
                   credit_history: str = 'all credits at this bank paid back duly',
                   purpose: str = '(vacation - does not exist?)', savings: str = '.. >= 1000 DM ',
                   present_emp_since: str = '.. >= 7 years',
                   personal_status_sex: str = 'female : divorced/separated/married', other_debtors: str = 'co-applicant',
                   property: str = 'if not A121 : building society savings agreement/ life insurance',
                   other_installment_plans: str = 'bank', housing: str = 'for free',
                   job: str = 'management/ self-employed/ highly qualified employee/ officer', telephone: str = 'none',
                   foreign_worker: str = 'no'):
    x = [duration_in_month, credit_amount, installment_as_income_perc, present_res_since, age, credits_this_bank,
         people_under_maintenance]

    if account_check_status == '0 <= ... < 200 DM':
        x.extend([1, 0, 0, 0])
    elif account_check_status == '< 0 DM':
        x.extend([0, 1, 0, 0])
    elif account_check_status == '>= 200 DM / salary assignments for at least 1 year':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])

    if credit_history == 'all credits at this bank paid back duly':
        x.extend([1, 0, 0, 0, 0])
    elif credit_history == 'critical account/ other credits existing (not at this bank)':
        x.extend([0, 1, 0, 0, 0])
    elif credit_history == 'delay in paying off in the past':
        x.extend([0, 0, 1, 0, 0])
    elif credit_history == 'existing credits paid back duly till now':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])

    if purpose == '(vacation - does not exist?)':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif purpose == 'business':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif purpose == 'car (new)':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif purpose == 'car (used)':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif purpose == 'domestic appliances':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif purpose == 'education':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif purpose == 'furniture/equipment':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif purpose == 'radio/television':
        x.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif purpose == 'repairs':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    if savings == '.. >= 1000 DM ':
        x.extend([1, 0, 0, 0, 0])
    elif savings == '... < 100 DM':
        x.extend([0, 1, 0, 0, 0])
    elif savings == '100 <= ... < 500 DM':
        x.extend([0, 0, 1, 0, 0])
    elif savings == '500 <= ... < 1000 DM ':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])

    if present_emp_since == '.. >= 7 years':
        x.extend([1, 0, 0, 0, 0])
    elif present_emp_since == '... < 1 year ':
        x.extend([0, 1, 0, 0, 0])
    elif present_emp_since == '1 <= ... < 4 years':
        x.extend([0, 0, 1, 0, 0])
    elif present_emp_since == '4 <= ... < 7 years':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])

    if personal_status_sex == 'female : divorced/separated/married':
        x.extend([1, 0, 0, 0])
    elif personal_status_sex == 'male : divorced/separated':
        x.extend([0, 1, 0, 0])
    elif personal_status_sex == 'male : married/widowed':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])

    if other_debtors == 'co-applicant':
        x.extend([1, 0, 0])
    elif other_debtors == 'guarantor':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])

    if property == 'if not A121 : building society savings agreement/ life insurance':
        x.extend([1, 0, 0, 0])
    elif property == 'if not A121/A122 : car or other, not in attribute 6':
        x.extend([0, 1, 0, 0])
    elif property == 'real estate':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])

    if other_installment_plans == 'bank':
        x.extend([1, 0, 0])
    elif other_installment_plans == 'none':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])

    if housing == 'for free':
        x.extend([1, 0, 0])
    elif housing == 'own':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])

    if job == 'management/ self-employed/ highly qualified employee/ officer':
        x.extend([1, 0, 0, 0])
    elif job == 'skilled employee / official':
        x.extend([0, 1, 0, 0])
    elif job == 'unemployed/ unskilled - non-resident':
        x.extend([0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 1])

    if telephone == 'none':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    if foreign_worker == 'no':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    x = np.array(x)

    explanation = explainer_object_german.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)

    write_explanation_to_file(explanation)

    return True


@app.get("/compas_lore")
def explain_compas(model_to_explain: str = 'RandomForestClassifier', age: int = 18, priors_count: int = 0, days_b_screening_arrest: int = 0, is_recid: int = 0,
                   is_violent_recid: int = 0, two_year_recid: int = 0, length_of_stay: int = 0,
                   age_cat: str = 'Less than 25', sex: str = 'Female', race: str = 'Caucasian',
                   c_charge_degree: str = 'M'):
    x = [age, priors_count, days_b_screening_arrest, is_recid, is_violent_recid, two_year_recid, length_of_stay]
    if age_cat == '25 - 45':
        x.extend([1, 0, 0])
    elif age_cat == 'Greater than 45':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])

    if sex == 'Female':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    if race == 'African-American':
        x.extend([1, 0, 0, 0, 0, 0])
    elif race == 'Asian':
        x.extend([0, 1, 0, 0, 0, 0])
    elif race == 'Caucasian':
        x.extend([0, 0, 1, 0, 0, 0])
    elif race == 'Hispanic':
        x.extend([0, 0, 0, 1, 0, 0])
    elif race == 'Native American':
        x.extend([0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 1])

    if c_charge_degree == 'F':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    x = np.array(x)

    blackbox = load_blackbox_compas(model_to_explain)

    class_compas = predict_class(x, blackbox, compas_class_values)
    class_prob = max(blackbox.predict_proba(x.reshape(1, -1))[0])

    return {'class_compas': class_compas,
            'class_prob': class_prob * 100,
            }


@app.get("/compas_lore_explanation")
def compas_lore_explanation(model_to_explain: str = 'RandomForestClassifier', age: int = 18, priors_count: int = 0, days_b_screening_arrest: int = 0, is_recid: int = 0,
                   is_violent_recid: int = 0, two_year_recid: int = 0, length_of_stay: int = 0,
                   age_cat: str = 'Less than 25', sex: str = 'Female', race: str = 'Caucasian',
                   c_charge_degree: str = 'M'):
    x = [age, priors_count, days_b_screening_arrest, is_recid, is_violent_recid, two_year_recid, length_of_stay]
    if age_cat == '25 - 45':
        x.extend([1, 0, 0])
    elif age_cat == 'Greater than 45':
        x.extend([0, 1, 0])
    else:
        x.extend([0, 0, 1])

    if sex == 'Female':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    if race == 'African-American':
        x.extend([1, 0, 0, 0, 0, 0])
    elif race == 'Asian':
        x.extend([0, 1, 0, 0, 0, 0])
    elif race == 'Caucasian':
        x.extend([0, 0, 1, 0, 0, 0])
    elif race == 'Hispanic':
        x.extend([0, 0, 0, 1, 0, 0])
    elif race == 'Native American':
        x.extend([0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 1])

    if c_charge_degree == 'F':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    x = np.array(x)

    explanation = explainer_object_compas.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)

    write_explanation_to_file(explanation)

    return True


@app.get("/adult_lore")
def explain_adult(model_to_explain: str = 'RandomForestClassifier', age: int = 17, capital_gain: int = 0, capital_loss: int = 0, hours_per_week: int = 0,
                   workclass: str = '', education: str = '', marital_status: str = '', occupation: str = '',
                    relationship: str = '', race: str = '', sex: str = '', native_country: str = ''):
    x = [age, capital_gain, capital_loss, hours_per_week]
    if workclass == 'Federal-gov':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0])
    elif workclass == 'Local-gov':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0])
    elif workclass == 'Never-worked':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0])
    elif workclass == 'Private':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0])
    elif workclass == 'Self-emp-inc':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0])
    elif workclass == 'Self-emp-not-inc':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0])
    elif workclass == 'State-gov':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 1])

    if education == '10th':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '11th':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '12th':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '1st-4th':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '5th-6th':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '7th-8th':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '9th':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == 'Assoc-acdm':
        x.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == 'Assoc-voc':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif education == 'Bachelors':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif education == 'Doctorate':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif education == 'HS-grad':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif education == 'Masters':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif education == 'Preschool':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif education == 'Prof-school':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    if marital_status == 'Divorced':
        x.extend([1, 0, 0, 0, 0, 0, 0])
    elif marital_status == 'Married-AF-spouse':
        x.extend([0, 1, 0, 0, 0, 0, 0])
    elif marital_status == 'Married-civ-spouse':
        x.extend([0, 0, 1, 0, 0, 0, 0])
    elif marital_status == 'Married-spouse-absent':
        x.extend([0, 0, 0, 1, 0, 0, 0])
    elif marital_status == 'Never-married':
        x.extend([0, 0, 0, 0, 1, 0, 0])
    elif marital_status == 'Separated':
        x.extend([0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 1])


    if occupation == 'Adm-clerical':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Armed-Forces':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Craft-repair':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Exec-managerial':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Farming-fishing':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Handlers-cleaners':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Machine-op-inspct':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Other-service':
        x.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Priv-house-serv':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif occupation == 'Prof-specialty':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif occupation == 'Protective-serv':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif occupation == 'Sales':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif occupation == 'Tech-support':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    if relationship == 'Husband':
        x.extend([1, 0, 0, 0, 0, 0])
    elif relationship == 'Not-in-family':
        x.extend([0, 1, 0, 0, 0, 0])
    elif relationship == 'Other-relative':
        x.extend([0, 0, 1, 0, 0, 0])
    elif relationship == 'Own-child':
        x.extend([0, 0, 0, 1, 0, 0])
    elif relationship == 'Unmarried':
        x.extend([0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 1])

    if race == 'Amer-Indian-Eskimo':
        x.extend([1, 0, 0, 0, 0])
    elif race == 'Asian-Pac-Islander':
        x.extend([0, 1, 0, 0, 0])
    elif race == 'Black':
        x.extend([0, 0, 1, 0, 0])
    elif race == 'Other':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])


    if sex == 'Female':
        x.extend([1, 0])
    else:
        x.extend([0, 1])


    if native_country == 'Cambodia':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Canada':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'China':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Columbia':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Cuba':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Dominican-Republic':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Ecuador':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'El-Salvador':
        x.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'England':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'France':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Germany':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Greece':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Guatemala':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Haiti':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Holand-Netherlands':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Honduras':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Hong':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Hungary':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'India':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Iran':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Ireland':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Italy':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Jamaica':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Japan':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Laos':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Mexico':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Nicaragua':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Outlying-US(Guam-USVI-etc)':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Peru':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Philippines':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Poland':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Portugal':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Puerto-Rico':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Scotland':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'South':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                  0, 0, 0, 0, 0, 0])
    elif native_country == 'Taiwan':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 0])
    elif native_country == 'Thailand':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0])
    elif native_country == 'Trinadad&Tobago':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 0])
    elif native_country == 'United-States':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0])
    elif native_country == 'Vietnam':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1])

    x = np.array(x)

    blackbox = load_blackbox_adult(model_to_explain)

    class_adult = predict_class(x, blackbox, adult_class_values)
    class_prob = max(blackbox.predict_proba(x.reshape(1, -1))[0])

    return {'class_adult': class_adult,
            'class_prob': class_prob * 100,
            }


@app.get("/adult_lore_explanation")
def adult_lore_explanation(model_to_explain: str = 'RandomForestClassifier', age: int = 17, capital_gain: int = 0, capital_loss: int = 0, hours_per_week: int = 0,
                   workclass: str = '', education: str = '', marital_status: str = '', occupation: str = '',
                    relationship: str = '', race: str = '', sex: str = '', native_country: str = ''):
    x = [age, capital_gain, capital_loss, hours_per_week]
    if workclass == 'Federal-gov':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0])
    elif workclass == 'Local-gov':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0])
    elif workclass == 'Never-worked':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0])
    elif workclass == 'Private':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0])
    elif workclass == 'Self-emp-inc':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0])
    elif workclass == 'Self-emp-not-inc':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0])
    elif workclass == 'State-gov':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 1])

    if education == '10th':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '11th':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '12th':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '1st-4th':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '5th-6th':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '7th-8th':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == '9th':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == 'Assoc-acdm':
        x.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif education == 'Assoc-voc':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif education == 'Bachelors':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif education == 'Doctorate':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif education == 'HS-grad':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif education == 'Masters':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif education == 'Preschool':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif education == 'Prof-school':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    if marital_status == 'Divorced':
        x.extend([1, 0, 0, 0, 0, 0, 0])
    elif marital_status == 'Married-AF-spouse':
        x.extend([0, 1, 0, 0, 0, 0, 0])
    elif marital_status == 'Married-civ-spouse':
        x.extend([0, 0, 1, 0, 0, 0, 0])
    elif marital_status == 'Married-spouse-absent':
        x.extend([0, 0, 0, 1, 0, 0, 0])
    elif marital_status == 'Never-married':
        x.extend([0, 0, 0, 0, 1, 0, 0])
    elif marital_status == 'Separated':
        x.extend([0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 1])

    if occupation == 'Adm-clerical':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Armed-Forces':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Craft-repair':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Exec-managerial':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Farming-fishing':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Handlers-cleaners':
        x.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Machine-op-inspct':
        x.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Other-service':
        x.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif occupation == 'Priv-house-serv':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif occupation == 'Prof-specialty':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif occupation == 'Protective-serv':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif occupation == 'Sales':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif occupation == 'Tech-support':
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    if relationship == 'Husband':
        x.extend([1, 0, 0, 0, 0, 0])
    elif relationship == 'Not-in-family':
        x.extend([0, 1, 0, 0, 0, 0])
    elif relationship == 'Other-relative':
        x.extend([0, 0, 1, 0, 0, 0])
    elif relationship == 'Own-child':
        x.extend([0, 0, 0, 1, 0, 0])
    elif relationship == 'Unmarried':
        x.extend([0, 0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 0, 1])

    if race == 'Amer-Indian-Eskimo':
        x.extend([1, 0, 0, 0, 0])
    elif race == 'Asian-Pac-Islander':
        x.extend([0, 1, 0, 0, 0])
    elif race == 'Black':
        x.extend([0, 0, 1, 0, 0])
    elif race == 'Other':
        x.extend([0, 0, 0, 1, 0])
    else:
        x.extend([0, 0, 0, 0, 1])

    if sex == 'Female':
        x.extend([1, 0])
    else:
        x.extend([0, 1])

    if native_country == 'Cambodia':
        x.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Canada':
        x.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'China':
        x.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Columbia':
        x.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Cuba':
        x.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    elif native_country == 'Dominican-Republic':
        x.extend(
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Ecuador':
        x.extend(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'El-Salvador':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'England':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'France':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Germany':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Greece':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Guatemala':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Haiti':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Holand-Netherlands':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Honduras':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Hong':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Hungary':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'India':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Iran':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Ireland':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Italy':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Jamaica':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Japan':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Laos':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Mexico':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Nicaragua':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Outlying-US(Guam-USVI-etc)':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Peru':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Philippines':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Poland':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Portugal':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Puerto-Rico':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Scotland':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'South':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0])
    elif native_country == 'Taiwan':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0])
    elif native_country == 'Thailand':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0])
    elif native_country == 'Trinadad&Tobago':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0])
    elif native_country == 'United-States':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0])
    elif native_country == 'Vietnam':
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0])
    else:
        x.extend(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1])

    x = np.array(x)

    explanation = explainer_object_adult.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)

    write_explanation_to_file(explanation)

    return True


@app.get("/wine_lore")
def explain_wine(model_to_explain: str = 'RandomForestClassifier', Alcohol: float = 0, Malic_acid: float = 0,
                          Ash: float = 0, Acl: float = 0, Mg: int = 0, Phenols: float = 0, Flavanoids: float = 0,
                          Nonflavanoid_phenols: float = 0, Proanth: float = 0, Color_int: float = 0, Hue: float = 0,
                          OD: float = 0, Proline: int = 0):
    x = [Alcohol,
        Malic_acid,
        Ash,
        Acl,
        Mg,
        Phenols,
        Flavanoids,
        Nonflavanoid_phenols,
        Proanth,
        Color_int,
        Hue,
        OD,
        Proline]
    x = np.array(x)

    blackbox = load_blackbox_wine(model_to_explain)

    class_wine = predict_class(x, blackbox, wine_class_values)
    class_prob = max(blackbox.predict_proba(x.reshape(1, -1))[0])

    return {'class_wine': class_wine,
            'class_prob': class_prob * 100,
            }



@app.get("/wine_lore_explanation")
def wine_lore_explanation(model_to_explain: str = 'RandomForestClassifier', Alcohol: float = 0, Malic_acid: float = 0,
                          Ash: float = 0, Acl: float = 0, Mg: int = 0, Phenols: float = 0, Flavanoids: float = 0,
                          Nonflavanoid_phenols: float = 0, Proanth: float = 0, Color_int: float = 0, Hue: float = 0,
                          OD: float = 0, Proline: int = 0):
    x = [Alcohol,
         Malic_acid,
         Ash,
         Acl,
         Mg,
         Phenols,
         Flavanoids,
         Nonflavanoid_phenols,
         Proanth,
         Color_int,
         Hue,
         OD,
         Proline]
    x = np.array(x)

    explanation = explainer_object_wine.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)

    write_explanation_to_file(explanation)

    return True


##########################################
# Get the Explanation from the Explainer #
##########################################
def get_explanation(intent, explanation):
    if intent == 'why':
        exp = "The factual rules are: \n" + explanation[0]
    elif intent == 'feature_importance':
        exp = "The most important features are: \n" + explanation[1]
    elif intent == 'how_to_be_that':
        exp = "The counterfactual rules are: \n" + explanation[2]
    elif intent == 'exemplar':
        exp = "The exemplars are: \n" + explanation[3]
    elif intent == 'counter_exemplar':
        exp = "The counter exemplars are: \n" + explanation[4]
    elif intent == 'performance':
        exp = "The fidelity is: \n" + explanation[5]
    else:
        exp = "Nothing found!"

    return exp



######################################################
# Version 1 - Bidirectional LSTM and Embedding Model #
######################################################
# @app.get("/chat_bot")
# def chat_bot(user_input: str = ""):
#
#     file = open("blackbox_model_files/verbatim_explanation.txt", "rt")
#     file_contents = file.read()
#     explanation = file_contents.split("\n\n")
#
#     output_response, context, intent = xbot_response.chatbot_response(user_input)
#
#     if intent in explanation_intents:
#         xbot_explanation = get_explanation(intent, explanation)
#     elif intent == 'greet':
#         xbot_explanation = random.choice(greet_response)
#     elif intent == 'goodbye':
#         xbot_explanation = random.choice(goodbye_response)
#     elif intent == 'bot_challenge':
#         xbot_explanation = random.choice(bot_challenge_response)
#     elif intent == 'options':
#         xbot_explanation = random.choice(options_response)
#     else:
#         xbot_explanation = "Nothing found!"
#
#     return {'tag_intent': intent,
#             'xbot_explanation': xbot_explanation}



##########################
# Version 2 - RASA Model #
##########################
@app.get("/chat_bot")
def chat_bot(user_input: str = ""):
    # with open('blackbox_model_files/explanation.json') as jsonfile:
    #     explanation = json.load(jsonfile)

    file = open("blackbox_model_files/verbatim_explanation.txt", "rt")
    file_contents = file.read()
    explanation = file_contents.split("\n\n")

    result = interpreter.parse(user_input, only_output_properties=False)

    intent = result.get('intent').get('name')
    if intent in explanation_intents:
        xbot_explanation = get_explanation(intent, explanation)
    elif intent == 'greet':
        xbot_explanation = random.choice(greet_response)
    elif intent == 'goodbye':
        xbot_explanation = random.choice(goodbye_response)
    elif intent == 'bot_challenge':
        xbot_explanation = random.choice(bot_challenge_response)
    elif intent == 'options':
        xbot_explanation = random.choice(options_response)
    else:
        xbot_explanation = "Nothing found!"

    return {'tag_intent': intent,
            'xbot_explanation': xbot_explanation}
