import nltk
import numpy as np
import random
import pickle

from util import record2str

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import re, string
punct = string.punctuation
trantab = str.maketrans(punct, len(punct) * ' ')  # Every punctuation symbol will be replaced by a space
whitelist = ['?', '!']

from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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


class ResponseGenerator(object):
    def __init__(self, data, train_x, train_y, tokenizer, model, explainer_data,
                 explainer_prepared_dataset, explainer_obj, blackbox, verbose=False):
        self.data = data
        self.train_x = train_x
        self.train_y = train_y
        self.tokenizer = tokenizer
        self.model = model
        self.explainer_data = explainer_data
        self.explainer_obj = explainer_obj
        self.blackbox = blackbox
        self.verbose = verbose

        self.feature_names = explainer_prepared_dataset[1]
        self.class_values = explainer_prepared_dataset[2]
        self.numeric_columns = explainer_prepared_dataset[3]

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

    def correct_spellings(self, text):
        """ For a given sentence this function returns a sentence after correcting
          spelling of words """
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0]._term

    def detokenize(self, text):
        return TreebankWordDetokenizer().detokenize(text)

    # return bag of words array: 0 or 1 for each word in the bag that exists in the user input
    def texts_to_sequences(self, user_input):
        # correcting spelling of words of user input
        user_input = self.correct_spellings(user_input)
        # tokenize the user input
        user_input_words = self.clean_text(user_input)
        detokenize_words = self.detokenize(user_input_words)

        # Transforms each text in texts to a sequence of integers.
        tokenized_seq = self.tokenizer.texts_to_sequences([detokenize_words])

        # get max length of tokenized sequential from training data
        max_length = len(self.train_x[0])
        # pad_sequences is used to ensure that all sequences in a list have the same length
        padded_sentences = pad_sequences(tokenized_seq, maxlen=max_length, padding='post')
        user_input_seq = np.array(padded_sentences)

        return user_input_seq

    def predict_class(self, user_input):
        return_list = []
        user_input_seq = self.texts_to_sequences(user_input)
        # check if all elements of user input sequential is zero or not
        if np.all((user_input_seq[0] == 0)):
            return_list.append({'intent': 'noanswer', 'probability': '1'})
        else:
            res = self.model.predict(user_input_seq)[0]
            # filter out predictions below a threshold
            ERROR_THRESHOLD = 0.6
            results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

            # sort by strength of probability
            results.sort(key=lambda x: x[1], reverse=True)

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
                tag_intent = i['tag']
                break
        return result, context, tag_intent

    def chatbot_response(self, user_input):
        # predict the intent of the user query
        intents = self.predict_class(user_input)
        # ask for the response
        output_response, context, tag_intent = self.get_response(intents)

        return output_response, context[0], tag_intent

    def get_explanation(self, context, explanation):
        if context == 'rule':
            exp = explanation.rule
        elif context == 'crule':
            exp = explanation.cstr()
        elif context == 'fidelity':
            exp = explanation.fidelity
        elif context == 'exemplar':
            exp = explanation.exemplars
        elif context == 'cexemplar':
            exp = explanation.cexemplars
        elif context == 'feature_importance':
            exp = explanation.feature_importance
        else:
            exp = "Nothing found!"

        return exp

    def start(self):
        print("Hello, I'm X-Bot!"
              "\n(If you want to exit, type -1!)")
        print("\nX-Bot: The length of the dataset is %s." % len(self.explainer_data))

        flag_exit = False
        flag_explainer = True
        while flag_explainer:
            print('\n')
            print("Select an instance x (enter a row number)")
            user_input = input()
            try:
                i = int(user_input)
            except:
                print("Please enter a valid number!")
                continue
            if i < 0:
                flag_exit = True
                print("See you later!")
                break
            elif i > len(self.explainer_data):
                print("It must be less than the length of the dataset (%s)!" % len(self.explainer_data))
                continue
            else:
                x = self.explainer_data[i]
                y_val_name = self.class_values[self.blackbox.predict(x.reshape(1, -1))[0]]
                print('\nX-Bot: Selected x is -> %s.' % record2str(x, self.feature_names, self.numeric_columns))
                print('Blackbox(x) = { %s }' % y_val_name)
                print('\n')

                print('X-Bot: Do you want me to continue with this instance? [y/n]')
                is_yes = input()
                try:
                    if is_yes == "y":
                        explanation = self.explainer_obj.explain_instance(x, samples=1000,
                                                                          nbr_runs=10, exemplar_num=3)
                        flag_explainer = False
                    else:
                        continue
                except:
                    print("Please enter y or n!")
                    continue

        flag_chat = True
        if not flag_exit:
            print("\n")
            print("X-Bot: Ok! How can I help you?"
                  "\n(If you want to exit, type Bye!)")
        while flag_chat and not flag_exit:

            user_input = input()
            if user_input == '':
                print('X-Bot: Please type something!\n')
            else:
                output_response, context, tag_intent = self.chatbot_response(user_input)

                print("Intent: %s" % tag_intent)
                print("X-Bot: %s" % output_response)
                if context != "" and context != "exit":
                    explain = self.get_explanation(context, explanation)
                    print(explain)
                    print('\n')
                elif context == "exit":
                    flag_chat = False
                else:
                    print('\n')