import re
import json

import pandas as pd
import numpy as np

from string import punctuation
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


class ToxicCommentClassification:

    def __init__(self):

        self.classes_ = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.new_features = ['count_sent', 'count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len']
        self.stop_words = set(stopwords.words('english'))
        self.contraction_dict = json.load(open('inputs/contraction_dict.json', 'r'))

    def export_data(self):

        self.train = pd.read_csv('./inputs/train.csv')
        self.test = pd.read_csv('inputs/test.csv')
        self.submission = pd.read_csv('inputs/sample_submission.csv')

    def feature_engineering(self, X):

        X['count_sent'] = X['comment_text'].apply(lambda x: len(sent_tokenize(x)))
        X['count_word'] = X['comment_text'].apply(lambda x: len(x.split()))
        X['count_unique_word'] = X['comment_text'].apply(lambda x: len(set(x.split())))
        X['count_letters'] = X['comment_text'].apply(lambda x: len(x))
        X['count_punctuations'] = X['comment_text'].apply(lambda x: len([c for c in x if c in punctuation]))
        X['count_words_upper'] = X['comment_text'].apply(lambda x: len([w for w in x.split() if w.isupper()]))
        X['count_words_title'] = X['comment_text'].apply(lambda x: len([w for w in x.split() if w.istitle()]))
        X['count_stopwords'] = X['comment_text'].apply(lambda x: len([w for w in x.lower().split() if w in self.stop_words]))
        X['mean_word_len'] = X['comment_text'].apply(lambda x: np.mean([len(w) for w in x.split()]))

    def tokenizer(self, sentences):
        lemmatizer = WordNetLemmatizer()
        tokens = sentences.lower().split()
        sentences = ' '.join([self.contraction_dict[token] if token in self.contraction_dict else token for token in tokens])
        sentences = re.sub(f'[{punctuation}“”¨«»®´·º½¾¿¡§£₤‘’]', ' ', sentences)
        sentences = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', sentences)
        sentences = re.sub('\[\[.*\]', ' ', sentences)
        sentences = re.sub(r'[0-9+]', ' ', sentences)
        tokens = word_tokenize(sentences.lower())
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def nbsvm_model(self):

        def _log_count_ratio(X, y):

            epsilon = 1.0
            p = np.log((epsilon + X[y == 1].sum(axis=0)) / np.linalg.norm(epsilon + X[y == 1].sum(axis=0), 1))
            q = np.log((epsilon + X[y == 0].sum(axis=0)) / np.linalg.norm(epsilon + X[y == 0].sum(axis=0), 1))
            r = p - q

            return r

        self.feature_engineering(self.train)
        self.feature_engineering(self.test)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=self.tokenizer)
        vectorizer.fit(self.train['comment_text'])

        X_train = hstack([vectorizer.transform(self.train['comment_text']), np.nan_to_num(self.train[self.new_features], copy=False)]).tocsr()
        X_test = hstack([vectorizer.transform(self.test['comment_text']), np.nan_to_num(self.test[self.new_features], copy=False)]).tocsr()

        preds = np.zeros((len(self.test), len(self.classes_)))

        for i, label in enumerate(self.classes_):
            print(f'----------{label}----------')
            r = _log_count_ratio(X_train, self.train[label])
            X = X_train.multiply(r)
            model = LogisticRegression(dual=False, max_iter=1000, random_state=42)
            model.fit(X, self.train[label])
            preds[:, i] = model.predict_proba(X_test.multiply(r))[:, 1]

        submission_id = pd.DataFrame({'id': self.submission['id']})
        submission = pd.concat([submission_id, pd.DataFrame(preds, columns=self.classes_)], axis=1)
        submission.to_csv('./outputs/4.csv', index=False)


if __name__ == '__main__':

    tcc = ToxicCommentClassification()
    tcc.export_data()
    tcc.nbsvm_model()