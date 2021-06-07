from fastapi import FastAPI

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
pkl_filename = 'blackbox_model_files/compas_model.pkl'
pkl_explainer_object = 'blackbox_model_files/compas_explainer_object.pkl'
iris_class_values = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
compas_class_values = ['High', 'Low', 'Medium']

# Load the Model back from file
with open(pkl_filename, 'rb') as file:
    blackbox = pickle.load(file)
with open(pkl_explainer_object, 'rb') as f:
    explainer_object = pickle.load(f)


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


@app.get("/explain_iris")
def explain_iris(sepal_length: int = 0, sepal_width: int = 0, petal_length: int = 0, petal_width: int = 0):
    x = [sepal_length, sepal_width, petal_length, petal_width]
    x = np.array(x)
    y_val = blackbox.predict(x.reshape(1,-1))[0]
    class_iris = iris_class_values[y_val]

    explanation = explainer_object.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)
    with open('blackbox_model_files/explanation.json', 'w') as jsonfile:
        json.dump(explanation, jsonfile, cls=ExplanationEncoder)

    verbatim_explanation = str(explanation)
    f = open("verbatim_explanation.txt", "w")
    f.write(verbatim_explanation)
    f.close()

    return {'class': class_iris,
            }

@app.get("/explain_compas")
def explain_compas(age: int = 0, priors_count: int = 0, days_b_screening_arrest: int = 0, is_recid: int = 0,
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
    y_val = blackbox.predict(x.reshape(1,-1))[0]
    class_compas = compas_class_values[y_val]

    explanation = explainer_object.explain_instance(x, samples=1000, nbr_runs=10, exemplar_num=3)
    with open('blackbox_model_files/explanation.json', 'w') as jsonfile:
        json.dump(explanation, jsonfile, cls=ExplanationEncoder)

    verbatim_explanation = str(explanation)
    f = open("blackbox_model_files/verbatim_explanation.txt", "w")
    f.write(verbatim_explanation)
    f.close()

    return {'class': class_compas,
            }


##########################################
# Get the Explanation from the Explainer #
##########################################
def get_explanation(intent, explanation):
    if intent == 'why':
        exp = "The factual rules are: " + explanation[0]
    elif intent == 'feature_importance':
        exp = "The most important features are: " + explanation[1]
    elif intent == 'how_to_be_that':
        exp = "The counterfactual rules are: " + explanation[2]
    elif intent == 'exemplar':
        exp = "The exemplars are: " + explanation[3]
    elif intent == 'counter_exemplar':
        exp = "The counter exemplars are: " + explanation[4]
    elif intent == 'performance':
        exp = "The fidelity is: " + explanation[5]
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
