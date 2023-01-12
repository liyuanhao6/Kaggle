import os
import re
import json
import time
import warnings
import random

import pandas as pd
import numpy as np

from string import punctuation

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")


def seed_setting(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class SpatialDropout(nn.Dropout2d):

    def forward(self, x):
        x = x.unsqueeze(2)                          # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)                   # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)                   # (N, T, 1, K)
        x = x.squeeze(2)                            # (N, T, K)

        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, max_features, num_aux_targets, lstm_units=128, dense_hidden_units=4*128):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)

        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


class Embedding():

    def __init__(self, embedding_file_path, word_index):
        self.embedding_file_path = embedding_file_path
        self.word_index = word_index
        self.embedding_matrix, self.unknown_words = self.build_matrix()

    def get_coefs(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def load_embeddings(self):
        with open(self.embedding_file_path) as file:
            return dict(self.get_coefs(*line.strip().split(' ')) for line in file)

    def build_matrix(self):
        embedding_index = self.load_embeddings()
        embedding_matrix = np.zeros((len(self.word_index) + 1, 300))
        unknown_words = []

        for word, i in self.word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                unknown_words.append(word)
        return embedding_matrix, unknown_words


class ToxicityClassification:

    def __init__(self, maxlen=220, num_models=2):
        self.maxlen = maxlen
        self.num_models = num_models

        self.train_data_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
        self.test_data_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
        self.sample_submission_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'
        self.glove_embedding_file_path = '../input/glove840b300dtxt/glove.840B.300d.txt'
        self.fasttext_embedding_file_path = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

        self.contraction_dict = json.load(open('../input/contraction-dict/contraction_dict.json', 'r'))

    def export_basic_data(self):
        self.train = pd.read_csv(self.train_data_path)
        self.test = pd.read_csv(self.test_data_path)
        self.submission = pd.read_csv(self.sample_submission_path)

    def preprocess(self, sentences):
        sentences = sentences.replace("’", "'")
        tokens = sentences.lower().split()
        sentences = ' '.join([self.contraction_dict[token] if token in self.contraction_dict else token for token in tokens])
        sentences = re.sub(r'[0-9+]', ' ', sentences)
        sentences = re.sub(r'[\U0001F600-\U0001F64F]', ' ', sentences)
        sentences = re.sub(r'[\U0001F300-\U0001F5FF]', ' ', sentences)
        sentences = re.sub(r'[\U0001F680-\U0001F6FF]', ' ', sentences)
        sentences = re.sub(r'[\U0001F1E0-\U0001F1FF]', ' ', sentences)
        sentences = re.sub(r'[\U00002702-\U000027B0]', ' ', sentences)
        sentences = re.sub(r'[\U000024C2-\U0001F251]', ' ', sentences)
        sentences = re.sub(f'[{punctuation + "’“"}]', ' ', sentences)

        return sentences

    def data_processing(self):
        X_train = self.train['comment_text'].apply(lambda x: self.preprocess(x))
        y_train = np.where(self.train['target'] >= 0.5, 1, 0)
        y_aux_train = self.train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
        X_test = self.test['comment_text'].apply(lambda x: self.preprocess(x))

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(X_train))
        X_train = tokenizer.texts_to_sequences(X_train)
        X_train = pad_sequences(X_train, maxlen=self.maxlen)
        X_test = tokenizer.texts_to_sequences(X_test)
        X_test = pad_sequences(X_test, maxlen=self.maxlen)

        glove_embedding = Embedding(embedding_file_path=self.glove_embedding_file_path, word_index=tokenizer.word_index)
        print('n unknown words (glove): ', len(glove_embedding.unknown_words))
        fasttext_embedding = Embedding(embedding_file_path=self.fasttext_embedding_file_path, word_index=tokenizer.word_index)
        print('n unknown words (fasttext): ', len(fasttext_embedding.unknown_words))
        self.embedding_matrix = np.concatenate([glove_embedding.embedding_matrix, fasttext_embedding.embedding_matrix], axis=-1)

        X_train_torch = torch.tensor(X_train, dtype=torch.long).cuda()
        X_test_torch = torch.tensor(X_test, dtype=torch.long).cuda()
        y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32).cuda()

        train_dataset = data.TensorDataset(X_train_torch, y_train_torch)
        test_dataset = data.TensorDataset(X_test_torch)

        max_features = len(tokenizer.word_index) + 1

        return train_dataset, test_dataset, X_train_torch, X_test_torch, y_train_torch, y_aux_train, max_features

    def LSTM(self):
        train_dataset, test_dataset, X_train_torch, X_test_torch, y_train_torch, y_aux_train, max_features = self.data_processing()

        all_test_preds = []

        for model_idx in range(self.num_models):
            seed_setting(model_idx)

            model = NeuralNet(embedding_matrix=self.embedding_matrix, max_features=max_features, num_aux_targets=y_aux_train.shape[-1])
            model.cuda()
            test_preds = self.train_model(model, train_dataset, test_dataset, output_dim=y_train_torch.shape[-1], loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
            all_test_preds.append(test_preds)

        submission = pd.DataFrame.from_dict({
            'id': self.test['id'],
            'prediction': np.mean(all_test_preds, axis=0)[:, 0]
        })

        submission.to_csv('submission.csv', index=False)

    def train_model(self, model, train, test, loss_fn, output_dim, lr=0.001, batch_size=512, n_epochs=4, enable_checkpoint_ensemble=True):
        param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
        optimizer = torch.optim.Adam(param_lrs, lr=lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
        all_test_preds = []
        checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

        for epoch in range(n_epochs):
            start_time = time.time()

            scheduler.step()

            model.train()
            avg_loss = 0.
            for data in train_loader:
                x_batch = data[:-1]
                y_batch = data[-1]

                y_pred = model(*x_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
            model.eval()
            test_preds = np.zeros((len(test), output_dim))

            for i, x_batch in enumerate(test_loader):
                y_pred = torch.sigmoid(model(*x_batch).detach().cpu()).numpy()

                test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

            all_test_preds.append(test_preds)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Epoch {epoch+1}/{n_epochs} \t loss={avg_loss:.4f} \t time={elapsed_time:.2f}s')

        if enable_checkpoint_ensemble:
            test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
        else:
            test_preds = all_test_preds[-1]

        return test_preds


if __name__ == '__main__':
    tc = ToxicityClassification()
    tc.export_basic_data()
    tc.LSTM()