from fastapi import FastAPI

import json
import lore
from explanation import ExplanationEncoder

import numpy as np
import pickle

# from pydantic import BaseModel, Field

import tensorflow as tf
from response_generator import ResponseGenerator


pkl_filename = 'blackbox_model_files/iris_model.pkl'
pkl_explainer_object = 'blackbox_model_files/iris_explainer_object.pkl'
iris_class_values = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# Load the Model back from file
with open(pkl_filename, 'rb') as file:
    blackbox = pickle.load(file)
with open(pkl_explainer_object, 'rb') as f:
    explainer_object = pickle.load(f)

# Load Chat_bot Files
data_file = open('chat_dataset/intents.json').read()
data = json.loads(data_file)

xbot_trained_model = tf.keras.models.load_model('chatbot_model_files/best_xbot_model.h5')
with open('chatbot_model_files/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
embedding_matrix = np.load('chatbot_model_files/embedding_matrix.npy')
training_dataset = np.load('chatbot_model_files/training_dataset.npy', allow_pickle=True)

train_x = list(training_dataset[:, 0])
train_y = list(training_dataset[:, 1])

xbot_response = ResponseGenerator(data, train_x, train_y, tokenizer, xbot_trained_model,
                                  explainer_object, verbose=False)

# FASTAPI
app = FastAPI()


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

    return {'class_iris': class_iris,
            # 'explanation': json.loads(json.dumps(explanation, cls=ExplanationEncoder)),
            'verbatim_explanation': str(explanation),
            }


@app.get("/chat_bot")
def chat_bot(user_input: str = ""):
    with open('blackbox_model_files/explanation.json') as jsonfile:
        explanation = json.load(jsonfile)

    output_response, context, tag_intent = xbot_response.chatbot_response(user_input)
    xbot_explanation = xbot_response.get_explanation(context, explanation)

    return {'tag_intent': tag_intent,
            'output_response': output_response,
            'xbot_explanation': xbot_explanation}
