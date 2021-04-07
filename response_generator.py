import pandas as pd
import re
import wikipedia as wk
import random
import warnings
from text_preprocessing import CleanText
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

class ResponseGenerator(object):
    def __init__(self, data, explanation):
        self.data = data
        self.explanation = explanation
        self.welcome_input = ("ciao", "hello", "hi", "greetings", "what's up", "hey", "buongiorno")
        self.welcome_response = ["ciao", "hi", "hey", "*nods*", "hi there", "hello", "buongiorno"]

        ct = CleanText()
        self.data['question_clean'] = ct.fit_transform(self.data.question)
        self.intents = data['intent'].unique()

    def welcome(self, user_input):
        for word in user_input.split():
            if word.lower() in self.welcome_input:
                return random.choice(self.welcome_response)

    def response(self, user_input):
        chatbot_response = ''

        # clean the input
        user_input_series = pd.Series(user_input)
        ct = CleanText()
        inp = ct.fit_transform(user_input_series)

        data_ques_lst = list(self.data['question_clean'])
        data_ques_lst.append(inp[0])

        TfidfVec = TfidfVectorizer()
        tfidf = TfidfVec.fit_transform(data_ques_lst)
        # vals = cosine_similarity(tfidf[-1], tfidf)
        vals = linear_kernel(tfidf[-1], tfidf)

        # get intent
        idx = vals.argsort()[0][-2]
        chatbot_intent = self.data['intent'][idx]

        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        if "wiki" in user_input:
            print("Checking Wikipedia")
            if user_input:
                chatbot_intent = 'wikipedia'
                chatbot_response = self.wikipedia_search(user_input)

        elif req_tfidf == 0:
            chatbot_intent = "Nothing found!"
            chatbot_response = "Nothing found!"

        else:
            if chatbot_intent == 'why':
                chatbot_response = self.explanation.rule
            elif chatbot_intent == 'how to be that':
                chatbot_response = self.explanation.cstr()
            elif chatbot_intent == 'performance':
                chatbot_response = self.explanation.fidelity
            elif chatbot_intent == 'exemplar':
                chatbot_response = self.explanation.exemplars
            elif chatbot_intent == 'counter exemplar':
                chatbot_response = self.explanation.cexemplars
            elif chatbot_intent == 'feature importance':
                chatbot_response = self.explanation.feature_importance
            else:
                chatbot_response = ""

        return chatbot_response, chatbot_intent

    def wikipedia_search(self, input):
        reg_ex = re.search('wiki (.*)', input)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                wiki = wk.summary(topic, sentences=3)
                return wiki
        except Exception as e:
            print("Nothing found!")

    # def counter_rules(self):
    #     deltas_str = '{ '
    #     for i, delta in enumerate(self.explanation.deltas):
    #         deltas_str += '{ ' if i > 0 else '{ '
    #         deltas_str += ', '.join([str(s) for s in delta])
    #         deltas_str += ' } --> %s, ' % self.explanation.crules[i]._cstr()
    #     deltas_str = deltas_str[:-2] + ' }'
    #     return deltas_str

    def start(self):
        flag = True
        print("Hello, I'm X-Bot! \nTo search a keyword in Wikipedia, write 'wiki + keyword'. "
              "\nIf you want to exit, type Bye!")
        while flag is True:
            user_input = input()
            user_input = user_input.lower()
            if user_input == '':
                print('Please type something!\n')

            elif user_input not in ['bye', 'shutdown', 'exit', 'quit']:
                if user_input == 'thanks' or user_input == 'thank you':
                    flag = False
                    print("X-Bot: You are welcome.")
                    print("X-Bot Intent:", 'thanks\n')
                else:
                    if self.welcome(user_input) is not None:
                        print("X-Bot: %s" % self.welcome(user_input))
                        print("X-Bot Intent:", 'greeting\n')
                    else:
                        chatbot_response, chatbot_intent = self.response(user_input)
                        print("X-Bot: %s" % chatbot_response)
                        print("X-Bot Intent: %s \n" % chatbot_intent)

            else:
                flag = False
                print("X-Bot: Bye! ")
                print("X-Bot Intent: \n", 'goodbye')