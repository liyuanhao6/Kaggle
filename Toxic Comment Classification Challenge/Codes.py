import os
import re
import json
import warnings

import pandas as pd
import numpy as np

from string import punctuation
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


class ToxicCommentClassification:

    def __init__(self):

        self.classes_ = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.new_features = ['count_sent', 'count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len']
        self.stop_words = set(stopwords.words('english'))
        self.contraction_dict = json.load(open('inputs/contraction_dict.json', 'r'))

    def export_basic_data(self):

        self.train = pd.read_csv('./inputs/train.csv')
        self.test = pd.read_csv('inputs/test.csv')
        self.submission = pd.read_csv('inputs/sample_submission.csv')

    def export_more_data(self, is_pseudo_label=False, is_translate=False):

        self.train = pd.read_csv('./inputs/train.csv')
        self.test = pd.read_csv('inputs/test.csv')
        self.submission = pd.read_csv('inputs/sample_submission.csv')

        if is_pseudo_label:
            test_with_pseudo_label = pd.read_csv('./extended_data/test_with_pseudo_label.csv')
            index_list = []
            for row in test_with_pseudo_label[self.classes_].astype(float).iterrows():
                for col in self.classes_:
                    if row[1][col] >= 0.01 and row[1][col] <= 0.99:
                        index_list.append(row[0])
                        break
            test_with_pseudo_label = pd.concat([self.test, test_with_pseudo_label[self.classes_]], axis=1)
            test_with_pseudo_label.drop(index=test_with_pseudo_label.index[index_list], inplace=True)
            test_with_pseudo_label_data = test_with_pseudo_label[self.classes_].apply(lambda x: x+0.5).astype(int)
            test_with_pseudo_label = pd.concat([test_with_pseudo_label[['id', 'comment_text']], test_with_pseudo_label_data], axis=1)
            self.train = pd.concat([self.train, test_with_pseudo_label], axis=0)

        if is_translate:
            for language in ['es', 'de', 'fr']:
                train_with_translate = pd.read_csv('./extended_data/train_' + language + '.csv')
                self.train = pd.concat([self.train, train_with_translate], axis=0)

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

        self.feature_engineering(self.train)
        self.feature_engineering(self.test)

        train_corpus = self.train['comment_text'].apply(lambda x: self.tokenizer(x))
        test_corpus = self.test['comment_text'].apply(lambda x: self.tokenizer(x))

        vectorizer_unigrams = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', min_df=1, strip_accents='unicode', stop_words='english')
        vectorizer_unigrams.fit(train_corpus)
        X_train_unigrams = vectorizer_unigrams.transform(train_corpus)
        X_test_unigrams = vectorizer_unigrams.transform(test_corpus)

        vectorizer_bigrams = TfidfVectorizer(ngram_range=(2, 2), analyzer='word', min_df=10, strip_accents='unicode', stop_words='english')
        vectorizer_bigrams.fit(train_corpus)
        X_train_bigrams = vectorizer_bigrams.transform(train_corpus)
        X_test_bigrams = vectorizer_bigrams.transform(test_corpus)

        vectorizer_trigrams = TfidfVectorizer(ngram_range=(3, 3), analyzer='word', min_df=10, strip_accents='unicode', stop_words='english')
        vectorizer_trigrams.fit(train_corpus)
        X_train_trigrams = vectorizer_trigrams.transform(train_corpus)
        X_test_trigrams = vectorizer_trigrams.transform(test_corpus)

        vectorizer_chargrams = TfidfVectorizer(ngram_range=(2, 5), analyzer='char', min_df=100, strip_accents='unicode', stop_words='english')
        vectorizer_chargrams.fit(train_corpus)
        X_train_chargrams = vectorizer_chargrams.transform(train_corpus)
        X_test_chargrams = vectorizer_chargrams.transform(test_corpus)

        X_train_indirect_features = np.nan_to_num(self.train[self.new_features], copy=False)
        X_test_indirect_features = np.nan_to_num(self.test[self.new_features], copy=False)

        X_train = hstack((X_train_unigrams, X_train_bigrams, X_train_trigrams, X_train_chargrams, X_train_indirect_features)).tocsr()
        X_test = hstack((X_test_unigrams, X_test_bigrams, X_test_trigrams, X_test_chargrams, X_test_indirect_features)).tocsr()

        return X_train, X_test

    def nbsvm_model(self, file_name):

        def _log_count_ratio(X, y):

            epsilon = 1.0
            p = np.log((epsilon + X[y == 1].sum(axis=0)) / np.linalg.norm(epsilon + X[y == 1].sum(axis=0), 1))
            q = np.log((epsilon + X[y == 0].sum(axis=0)) / np.linalg.norm(epsilon + X[y == 0].sum(axis=0), 1))
            r = p - q

            return r

        X_train, X_test = self.data_processing()

        preds = np.zeros((len(self.test), len(self.classes_)))

        for i, label in enumerate(self.classes_):
            print(f'----------{label}----------')
            r = _log_count_ratio(X_train, self.train[label])
            X = X_train.multiply(r)
            model = LogisticRegression(dual=False, max_iter=2000, random_state=42)
            model.fit(X, self.train[label])
            preds[:, i] = model.predict_proba(X_test.multiply(r))[:, 1]

        submission_id = pd.DataFrame({'id': self.submission['id']})
        submission = pd.concat([submission_id, pd.DataFrame(preds, columns=self.classes_)], axis=1)
        submission.to_csv(file_name, index=False)


def baseline_nbsvm():
    tcc = ToxicCommentClassification()
    tcc.export_basic_data()
    tcc.nbsvm_model('./outputs/5.csv')


def nbsvm_with_pseudo_label(loop_num=5):
    if not os.path.exists('./extended_data'):
        os.mkdir('./extended_data')
    tcc = ToxicCommentClassification()
    tcc.export_basic_data()
    for i in range(loop_num):
        tcc.nbsvm_model('./extended_data/test_with_pseudo_label.csv')
        tcc.export_more_data(is_pseudo_label=True)
    tcc.nbsvm_model('./outputs/6.csv')


def nbsvm_with_data_augment():
    tcc = ToxicCommentClassification()
    tcc.export_more_data(is_translate=True)
    tcc.nbsvm_model('./outputs/7.csv')


if __name__ == '__main__':
    nbsvm_with_data_augment()
