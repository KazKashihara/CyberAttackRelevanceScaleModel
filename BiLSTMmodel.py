import os
import pandas as pd
import torch
import time
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import scikitplot as skplt

from torchmetrics.functional import f1

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer

from sklearn.preprocessing import StandardScaler
from multiprocessing import  Pool
from functools import partial
import numpy as np
from sklearn.decomposition import PCA
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def clean_text(x):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '',x)
    return text

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def create_data():
    # Read data
    df = pd.read_csv("./Data/Data.csv")
    labels = df['Score'].tolist()
    posts = df['Post Content'].tolist()

    datasets = pd.DataFrame(columns=["post", "score"])
    for i in range(len(labels)):
        post = posts[i]
        score = labels[i]
        s = pd.Series([post, score], index=datasets.columns)
        datasets = datasets.append(s, ignore_index=True)

    # Randomise the data
    datasets = datasets.sample(frac=1).reset_index(drop=True)
    datasets.head()

    train, test = train_test_split(datasets, test_size=0.2)

    train.to_csv('./Data/trainData.csv', index=False)
    test.to_csv('./Data/testData.csv', index=False)

    return train, test


## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule

def load_glove(word_index):
    EMBEDDING_FILE = 'Data/glove/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf-8"))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

#Pytorch Model - TextCNN
class CNN_Text(nn.Module):

    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [1, 2, 3, 5]
        num_filters = 36
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit



def plot_graph(epochs, filename):
    fig = plt.figure(figsize=(12,12))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    fig.savefig(filename)


class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        # rint(x.size())
        h_embedding = self.embedding(x)
        # _embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

def predict_single(x):
    # tokenize
    x = tokenizer.texts_to_sequences([x])
    # pad
    x = pad_sequences(x, maxlen=maxlen)
    # create dataset
    x = torch.tensor(x, dtype=torch.long).cuda()

    pred = model(x).detach()
    pred = F.softmax(pred).cpu().numpy()

    pred = pred.argmax(axis=1)

    pred = le.classes_[pred]
    return pred[0]

if __name__ == '__main__':


    #train, test = create_data()
    train = pd.read_csv("./Data/trainData.csv", encoding="utf-8")
    test = pd.read_csv("./Data/testData.csv", encoding="utf-8")


    # Bacis parameters
    embed_size = 300  # how big is each vector
    max_features = 12000  # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 750  # max number of words in a question to use
    batch_size = 512  # how many samples to process at once
    n_epochs = 10  # how many times to iterate over all samples
    n_splits = 5  # Number of K-fold Splits
    SEED = 10
    debug = 0

    train_X = train["post"].tolist()
    test_X_Original = test["post"].tolist()

    train_y_Original = train["score"].tolist()
    test_y_Original = test["score"].tolist()

    test_E = test["Exploit"].tolist()

    data = pd.concat([train, test])[['post', 'score']]

    # Finding the maxlen
    data['len'] = data['post'].apply(lambda s: len(s))
    data['len'].plot.hist(bins=100).figure.savefig('./Length.png')

    # maxlen = data.len.quantile(0.9)
    # print(maxlen)

    # tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X_Original)

    # pad the sentence
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)



    le = LabelEncoder()
    # labels = [0, 1, 2, 3]
    train_y = le.fit_transform(train_y_Original)
    test_y = le.transform(test_y_Original)
    # train_y = le.fit_transform(labels)
    # test_y = le.transform(labels)

    # missing entries in the embedding are set using np.random.normal so we have to seed here too

    if debug:
        embedding_matrix = np.random.randn(120000, 300)
    else:
        embedding_matrix = load_glove(tokenizer.word_index)

    np.shape(embedding_matrix)

    n_epochs = 90
    model = CNN_Text()
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    model.cuda()

    # Load train and test in CUDA Memory
    x_train = torch.tensor(train_X, dtype=torch.long).cuda()
    y_train = torch.tensor(train_y, dtype=torch.long).cuda()
    x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
    y_cv = torch.tensor(test_y, dtype=torch.long).cuda()

    # Create Torch datasets
    print(x_train.size(0))
    print(y_train.size(0))
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    train_loss = []
    valid_loss = []

    # Bi-LSTM
    n_epochs = 100
    model = BiLSTM()
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    model.cuda()

    # Load train and test in CUDA Memory
    x_train = torch.tensor(train_X, dtype=torch.long).cuda()
    y_train = torch.tensor(train_y, dtype=torch.long).cuda()
    x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
    y_cv = torch.tensor(test_y, dtype=torch.long).cuda()

    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    train_loss = []
    valid_loss = []

    for epoch in range(n_epochs):
        start_time = time.time()
        # Set model to train configuration
        model.train()
        avg_loss = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Predict/Forward Pass
            y_pred = model(x_batch)
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        # Set model to validation configuration -Doesn't get trained here
        model.eval()
        avg_val_loss = 0.
        val_preds = np.zeros((len(x_cv), len(le.classes_)))

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            # keep/store predictions
            val_preds[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred).cpu().numpy()

        # Check Accuracy
        val_accuracy = sum(val_preds.argmax(axis=1) == test_y) / len(test_y)
        train_loss.append(avg_loss)
        valid_loss.append(avg_val_loss)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))

    plot_graph(n_epochs, './BiLSTM-Epoch.png')
    torch.save(model, 'bilstm_model')
    y_true = [le.classes_[x] for x in test_y]
    y_pred = [le.classes_[x] for x in val_preds.argmax(axis=1)]
    labelInfo = ["0", "1", "2", "3"]

    conf_mat = confusion_matrix(test_y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, cmap="Blues", annot=True, fmt='d',
                xticklabels=labelInfo, yticklabels=labelInfo, linewidths=1, linecolor='black')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    outfile = "./BiLSTM-Performance.png"
    plt.savefig(outfile)

    skplt.metrics.plot_confusion_matrix(
        y_true,
        y_pred,
        figsize=(12, 12), x_tick_rotation=90).figure.savefig('./BiLSTM-Result.png')

    f1score = f1(torch.tensor(y_pred), torch.tensor(y_true), num_classes=4)
    print("BiLSTMScore:" + str(f1score))
    print(classification_report(y_true, y_pred))

    resultList = []
    for i in range(len(test_X_Original)):
        post = test_X_Original[i]
        #print(post)
        pairList = []
        pairList.append(post)
        pairList.append(predict_single(post))
        pairList.append(test_y[i])
        pairList.append(test_E[i])
        resultList.append(pairList)

    postSaveFile = "./Data/BiLSTM-Evaluation-Result.csv"
    S1 = pd.DataFrame(resultList, columns=['Post Content', 'Predicted Score', 'Score', 'Exploit'])
    S1.to_csv(postSaveFile)
