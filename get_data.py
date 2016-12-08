import numpy as np
from gensim.models import Word2Vec
from keras.utils import np_utils


MAX_LEN = 79
WORD_DIM = 50
sentences_train_X = []
sentences_test_X = []
sentences_train_Y = []
sentences_test_Y = []
model = Word2Vec.load("./model/50features_1minwords_10context")
POS_DIC = {"X":0,"O":1,"B":2,"I":3}
DIC_LEN = len(POS_DIC)

def get_basic_data(sentences_X,sentences_Y,filepath):
    f = open(filepath)
    list_X = []
    list_Y = []
    for line in f.readlines():
        if(line.strip()=="<end_for_sentence>"):
            sentences_X.append(list_X)
            sentences_Y.append(list_Y)
            list_X = []
            list_Y = []
        else:
            list_X.append(line.split(" ")[0].strip().lower())
            list_Y.append(line.split(" ")[1].strip())


def return_data():
    get_basic_data(sentences_test_X,sentences_test_Y,'./data/test')
    get_basic_data(sentences_train_X,sentences_train_Y,'./data/train')

    train_size = len(sentences_train_X)  # 2737
    test_size = len(sentences_test_X)  # 304

    train_X = np.zeros((train_size, MAX_LEN, WORD_DIM))
    train_Y = np.zeros((train_size, MAX_LEN,DIC_LEN))
    test_X = np.zeros((test_size, MAX_LEN, WORD_DIM))
    test_Y = np.zeros((test_size, MAX_LEN,DIC_LEN))

    for i,line in enumerate(sentences_train_X):
        for j,word in enumerate(line):
            train_X[i][j] = model[word]
            train_Y[i][j][POS_DIC[sentences_train_Y[i][j]]] = 1
    for i,line in enumerate(sentences_test_X):
        for j,word in enumerate(line):
            test_X[i][j] = model[word]
            test_Y[i][j][POS_DIC[sentences_test_Y[i][j]]] = 1

    for i in range(train_Y.shape[0]):
        for j in range(MAX_LEN):
            if train_Y[i][j][1]!=1 and train_Y[i][j][2]!=1 and train_Y[i][j][3]!=1:
                train_Y[i][j][0] = 1
    for i in range(test_Y.shape[0]):
         for j in range(MAX_LEN):
            if test_Y[i][j][1] != 1 and test_Y[i][j][2] != 1 and test_Y[i][j][3] != 1:
                test_Y[i][j][0] = 1
    return train_X,train_Y,test_X,test_Y



