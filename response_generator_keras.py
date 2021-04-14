import nltk
import numpy as np
import random
import pickle

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import re, string
punct = string.punctuation
trantab = str.maketrans(punct, len(punct) * ' ')  # Every punctuation symbol will be replaced by a space
whitelist = ['?', '!']

# Using symspell to correct spelling
import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy",
                                                  "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy",
                                              "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


class ResponseGeneratorKeras(object):
    def __init__(self, data, explanation, model, verbose=False):
        self.data = data
        self.explanation = explanation
        self.model = model
        self.verbose = verbose

        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))

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

    def text_clean(self, text):
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

    def correct_spellings(self, text):
        """ For a given sentence this function returns a sentence after correcting
          spelling of words """
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0]._term

    # return bag of words array: 0 or 1 for each word in the bag that exists in the user input
    def bag_of_words(self, user_input):
        # correcting spelling of words of user input
        user_input = self.correct_spellings(user_input)
        # tokenize the user input
        user_input_words = self.text_clean(user_input)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(self.words)
        for s in user_input_words:
            for i, w in enumerate(self.words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if self.verbose:
                        print("Found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, user_input):
        # filter out predictions below a threshold
        bag = self.bag_of_words(user_input)
        res = self.model.predict(np.array([bag]))[0]
        ERROR_THRESHOLD = 0.5
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, ints):
        result = ''
        context = ''
        if not ints:
            tag = 'noanswer'
        else:
            tag = ints[0]['intent']

        list_of_intents = self.data['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                context = i['context']
                break
        return result, context

    def chatbot_response(self, user_input):
        intents = self.predict_class(user_input)
        output_response, context = self.get_response(intents)

        return output_response, context[0]

    def get_explanation(self, context):
        if context == 'rule':
            exp = self.explanation.rule
        elif context == 'crule':
            exp = self.explanation.cstr()
        elif context == 'fidelity':
            exp = self.explanation.fidelity
        elif context == 'exemplar':
            exp = self.explanation.exemplars
        elif context == 'cexemplar':
            exp = self.explanation.cexemplars
        elif context == 'feature_importance':
            exp = self.explanation.feature_importance
        else:
            exp = "Nothing found!"

        return exp

    def start(self):
        flag = True
        print("Hello, I'm X-Bot!"
              "\nIf you want to exit, type Bye!")
        while flag is True:
            user_input = input()
            if user_input == '':
                print('X-Bot: Please type something!\n')
            else:
                output_response, context = self.chatbot_response(user_input)

                print("X-Bot: %s" % output_response)
                if context != "" and context != "exit":
                    explain = self.get_explanation(context)
                    print(explain)
                    print('\n')
                elif context == "exit":
                    flag = False
                else:
                    print('\n')