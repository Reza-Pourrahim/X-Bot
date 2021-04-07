'''
    text cleaning:

    remove the mentions
    remove the hash tag sign (#) but not the actual tag as this may contain information
    set all words to lowercase
    remove all punctuations, including the question and exclamation marks
    remove the urls as they do not contain useful information and we did not notice a distinction in the number of urls used between the sentiment classes
    make sure the converted emojis are kept as one word.
    remove digits
    remove stopwords
    apply the porterstemmer to keep the stem of the words
    apply lemmatizer
'''


from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
stop = stopwords.words('english')

import re, string



class CleanText(BaseEstimator, TransformerMixin):

    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)

    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_', '')

    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct) * ' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)

    def to_lower(self, input_text):
        return input_text.lower()

    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        whitelist = ["n't", "not", "no","how","who",'when','what','why','which','where','whose']
        words = input_text.split()
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
        return " ".join(clean_words)

    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split()
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)

    def fit(self, X, y=None, **fit_params):
        return self

    # def tokenize(self, input_text):
    #     tokenized_words = word_tokenize(input_text)
    #     return " ".join(tokenized_words)

    def tokenize_lemmatizer(self, input_text):
        tokenized_words = word_tokenize(input_text)
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        lmtzr = WordNetLemmatizer()
        lemma_list = []
        # rmv = [i for i in input_text if i]
        for token, tag in nltk.pos_tag(tokenized_words):
            lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
            lemma_list.append(lemma)
        return " ".join(lemma_list)

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(
            self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(
            self.stemming).apply(self.tokenize_lemmatizer)
        return clean_X