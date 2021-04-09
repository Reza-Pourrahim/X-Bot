import nltk
import numpy as np
import random
import pickle

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


class ResponseGeneratorKeras(object):
    def __init__(self, data, explanation, model, verbose=False):
        self.data = data
        self.explanation = explanation
        self.model = model
        self.verbose = verbose

        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))

    def clean_up_text(self, user_input):
        text_words = nltk.word_tokenize(user_input)
        text_words = [lemmatizer.lemmatize(word.lower()) for word in text_words]
        return text_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the user input
    def bag_of_words(self, user_input):
        # tokenize the pattern
        user_input_words = self.clean_up_text(user_input)
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
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints):
        result = ''
        context = ''
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
        output_response, context = self.getResponse(intents)

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
                print('Please type something!\n')

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