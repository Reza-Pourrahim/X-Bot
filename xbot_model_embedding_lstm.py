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
import re
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

class XBotModel(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def decontractions(self, phrase):
        """decontracted takes text and convert contractions into natural form.
         ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"won\’t", "will not", phrase)
        phrase = re.sub(r"can\’t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)

        phrase = re.sub(r"n\’t", " not", phrase)
        phrase = re.sub(r"\’re", " are", phrase)
        phrase = re.sub(r"\’s", " is", phrase)
        phrase = re.sub(r"\’d", " would", phrase)
        phrase = re.sub(r"\’ll", " will", phrase)
        phrase = re.sub(r"\’t", " not", phrase)
        phrase = re.sub(r"\’ve", " have", phrase)
        phrase = re.sub(r"\’m", " am", phrase)

        return phrase

    def clean_text(self, text):
        # Lower casting
        text = text.lower()
        # Decontracted
        text = self.decontractions(text)
        # Remove more than 1 space
        text = " ".join(text.split())
        # Remove punctuation
        text = text.translate(trantab)
        text_words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(w) for w in text_words if w not in whitelist]
        return words

    def detokenize(self, text):
        return TreebankWordDetokenizer().detokenize(text)


    def prepare_train_dataset(self, data):
        classes = []
        pattern_list = []
        pattern_class = []

        # adding classes to our class list
        # pattern_class = [intent['tag'] for intent in data['intents'] if intent not in pattern_class]

        for intent in data['intents']:
            for pattern in intent['patterns']:

                # clean the pattern
                w = self.clean_text(pattern)

                # adding documents
                detokenize_words = self.detokenize(w)
                pattern_list.append((detokenize_words))
                pattern_class.append(intent['tag'])



        # This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(pattern_list)
        # Transforms each text in texts to a sequence of integers.
        tokenized_seq = tokenizer.texts_to_sequences(pattern_list)

        # Determining the maximum length of the tokenized sequential
        max_length = max([len(s) for s in tokenized_seq])

        # pad_sequences is used to ensure that all sequences in a list have the same length
        padded_sentences = pad_sequences(tokenized_seq, maxlen=max_length, padding='post')
        data_pattern = list(padded_sentences)

        # labels (classes)
        labels = pd.get_dummies(pattern_class).values.tolist()

        # training dataset
        training_dataset = [[a]+[b] for a, b in zip(data_pattern, labels)]

        # shuffle our features and turn into np.array
        random.shuffle(training_dataset)
        training_dataset = np.array(training_dataset)


        # a list of different types of intents of responses
        classes = sorted(list(set(pattern_class)))
        # save as a pickle
        pickle.dump(classes, open('classes.pkl', 'wb'))

        # define vocabulary size (largest integer value)
        # The vocabulary size is the total number of words in our vocabulary, plus one for unknown words
        vocab_size = len(tokenizer.word_index) + 1
        if self.verbose:
            print("Training Data Created!")
            print("Words : {}".format(tokenizer.word_index))
            print('Number of  Words = {}'.format(vocab_size))
            print('Maximum Length of Words in a Sentence = {}'.format(max_length))

        return training_dataset, vocab_size, tokenizer


    def create_model(self, train_x, train_y, vocab_size=1000, output_dim=128, lstm_out=100, dropout=0.5):
        input_length = len(train_x[0])
        num_classes = len(train_y[0])

        # Model
        model = Sequential()
        # The embedding layer
        model.add(layers.Embedding(input_dim=vocab_size,
                                   output_dim=output_dim,
                                   input_length=input_length))
        # The Bidirectional LSTM layer
        model.add(layers.Bidirectional(layers.LSTM(lstm_out, dropout=dropout)))
        # The Dense layer
        model.add(layers.Dense(num_classes, activation='softmax'))

        if self.verbose:
            print(model.summary())

        return model

    def compile_fit_model(self, model, train_x, train_y, epochs=100, batch_size=5,
                          earlystopping_patience=10, validation_split=0.3, loss='categorical_crossentropy'):

        if self.verbose:
            verb = 1
        else:
            verb = 0

        es = EarlyStopping(monitor='loss', patience=earlystopping_patience, verbose=verb)
        mc = ModelCheckpoint('best_xbot_model.h5', monitor='val_accuracy', save_best_only=True)

        # Model compilation
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

        # Model Training and Validation
        mymodel = model.fit(x=np.array(train_x),
                            y=np.array(train_y),
                            validation_split=validation_split,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verb,
                            callbacks=[es, mc]).history
        return mymodel