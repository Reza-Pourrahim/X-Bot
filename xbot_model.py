import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import string
punct = string.punctuation
trantab = str.maketrans(punct, len(punct) * ' ')  # Every punctuation symbol will be replaced by a space
whitelist = ['?', '!']

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint



class XBotModel(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def text_clean(self, text):
        text = text.lower().translate(trantab) # remove_punctuation
        text_words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(w) for w in text_words if w not in whitelist]
        return words

    def prepare_train_dataset(self, data):
        words = []
        classes = []
        documents = []
        for intent in data['intents']:
            for pattern in intent['patterns']:

                # clean the pattern
                w = nltk.word_tokenize(pattern)
                w = self.text_clean(pattern)
                words.extend(w)

                # adding documents
                documents.append((w, intent['tag']))

                # adding classes to our class list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        # a list of different words that could be used for pattern recognition
        words = sorted(list(set(words)))
        # a list of different types of intents of responses
        classes = sorted(list(set(classes)))
        # save as a pickle
        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))

        # training dataset
        training_dataset = []
        output_empty = [0] * len(classes)

        for doc in documents:
            # initializing bag of words
            bag = []

            pattern_words = doc[0]
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

        # Model
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
                          earlystopping_patience=10, loss='categorical_crossentropy'):

        if self.verbose:
            verb = 1
        else:
            verb = 0

        es = EarlyStopping(monitor='val_loss', patience=earlystopping_patience, verbose=verb)
        mc = ModelCheckpoint('best_xbot_model.h5', monitor='val_loss', save_best_only=True)

        # Model compilation
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss=loss,
                      metrics=['accuracy'])

        # Model Training and Validation
        mymodel = model.fit(x=np.array(train_x),
                            y=np.array(train_y),
                            validation_split=0.3,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verb,
                            callbacks=[es, mc]).history
        return mymodel

