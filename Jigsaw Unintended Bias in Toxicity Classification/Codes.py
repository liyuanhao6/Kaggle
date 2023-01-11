import re
import json
import warnings

import pandas as pd
import numpy as np

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


class ToxicityClassification:

    def __init__(self):
        self.train_data_path = './inputs/train.csv'
        self.test_data_path = './inputs/test.csv'
        self.sample_submission_path = './inputs/sample_submission.csv'

        self.new_features = ['count_sent', 'count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len']
        self.stop_words = set(stopwords.words('english'))
        self.contraction_dict = json.load(open('./inputs/contraction_dict.json', 'r'))

    def export_basic_data(self):
        self.train = pd.read_csv(self.train_data_path)
        self.test = pd.read_csv(self.test_data_path)
        self.submission = pd.read_csv(self.sample_submission_path)

    def tokenizer(self, sentences):
        lemmatizer = WordNetLemmatizer()

        tokens = sentences.lower().split()
        sentences = ' '.join([self.contraction_dict[token] if token in self.contraction_dict else token for token in tokens])
        sentences = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', sentences)
        sentences = re.sub('https?://\S+|www\.\S+', ' ', sentences)
        sentences = re.sub('[^\x00-\x7f]', ' ', sentences)
        sentences = re.sub('\[\[.*\]', ' ', sentences)
        sentences = re.sub(r'[0-9+]', ' ', sentences)
        sentences = re.sub(r'[\U0001F600-\U0001F64F]', ' ', sentences)
        sentences = re.sub(r'[\U0001F300-\U0001F5FF]', ' ', sentences)
        sentences = re.sub(r'[\U0001F680-\U0001F6FF]', ' ', sentences)
        sentences = re.sub(r'[\U0001F1E0-\U0001F1FF]', ' ', sentences)
        sentences = re.sub(r'[\U00002702-\U000027B0]', ' ', sentences)
        sentences = re.sub(r'[\U000024C2-\U0001F251]', ' ', sentences)
        sentences = re.sub(f'[{punctuation}]', ' ', sentences)
        tokens = word_tokenize(sentences.lower())
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    def data_processing(self):

        train_corpus = self.train['comment_text'].apply(lambda x: self.tokenizer(x))
        test_corpus = self.test['comment_text'].apply(lambda x: self.tokenizer(x))

        vectorizer_unigrams = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', min_df=3, strip_accents='unicode', stop_words='english')
        vectorizer_unigrams.fit(train_corpus)

        X_train = vectorizer_unigrams.transform(train_corpus)
        X_test = vectorizer_unigrams.transform(test_corpus)
        y_train = np.where(self.train['target'] >= 0.5, 1., 0.)

        return X_train, X_test, y_train

    def lr_model(self, file_name):
        X_train, X_test, y_train = self.data_processing()

        model = LogisticRegression(dual=False, max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = np.where(model.predict(X_test) >= 0.5, 1., 0.)

        self.submission['prediction'] = y_pred
        self.submission.to_csv(file_name, index=False)


if __name__ == '__main__':
    tc = ToxicityClassification()
    tc.export_basic_data()
    tc.lr_model('./outputs/1.csv')