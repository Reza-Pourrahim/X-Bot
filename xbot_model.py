import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam


class XBotModel(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def prepare_train_dataset(self, data):
        words = []
        classes = []
        documents = []
        for intent in data['intents']:
            for pattern in intent['patterns']:
                # take each word and tokenize it
                w = nltk.word_tokenize(pattern)
                words.extend(w)

                # adding documents
                documents.append((w, intent['tag']))

                # adding classes to our class list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        ignore_words = ['?', '!']
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        # save as a pickle
        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))

        # train dataset
        training_dataset = []
        output_empty = [0] * len(classes)

        for doc in documents:
            # initializing bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training_dataset.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training_dataset)
        training_dataset = np.array(training_dataset)

        if self.verbose:
            print("Training data created!")

        return training_dataset

    def create_model(self, train_x, train_y, dropout=0.5):
        input_size = len(train_x[0])
        num_classes = len(train_y[0])

        # Model architecture
        model = Sequential([
            Dense(units=128, input_shape=(input_size,), activation='relu'),
            Dropout(dropout),
            Dense(units=64, activation='relu'),
            Dropout(dropout),
            Dense(units=num_classes, activation='softmax')
        ])
        if self.verbose:
            print(model.summary())

        return model

    def compile_fit_model(self, model, train_x, train_y, epochs=100, batch_size=5, lr=5e-3,
                          loss='categorical_crossentropy'):

        # Model compilation
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss=loss,
                      metrics=['accuracy'])

        # Model Training and Validation
        mymodel = model.fit(x=np.array(train_x),
                            y=np.array(train_y),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=1)
        return mymodel

