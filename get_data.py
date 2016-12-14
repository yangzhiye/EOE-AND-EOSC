import numpy as np
from gensim.models import Word2Vec


MAX_LEN = 79
WORD_DIM = 50
sentences_train_X = []
sentences_test_X = []
sentences_train_Y = []
sentences_test_Y = []
w2cmodel = Word2Vec.load("./model/50features_1minwords_10context")
POS_DIC = {"X":0,"O":1,"B":2,"I":3}
DIC_LEN = len(POS_DIC)

def get_basic_data(sentences_X,sentences_Y,filepath):
    f = open(filepath)
    list_X = []
    list_Y = []
    list_esc = []
    for line in f.readlines():
        if(line.strip()=="<end_for_sentence>"):
            sentences_X.append(list_X)
            sentences_Y.append(list_Y)
            list_X = []
            list_Y = []
        else:
            list_X.append(line.split(" ")[0].strip().lower())
            list_Y.append(line.split(" ")[1].strip())
            if line.split(" ")[1].strip() == 'B':
                list_esc.append(line.split(" ")[2].strip())
    return list_esc


def return_eoa_data():
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
            train_X[i][j] = w2cmodel[word]
            train_Y[i][j][POS_DIC[sentences_train_Y[i][j]]] = 1
    for i,line in enumerate(sentences_test_X):
        for j,word in enumerate(line):
            test_X[i][j] = w2cmodel[word]
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


def return_b_size(Y):
    size = 0
    for i in Y:
        for j in i:
            if j == 'B':
                size+=1
    return size


def return_max_sc(test_Y,train_Y):
    max_sc = 0
    for i,line in enumerate(test_Y):
        for j,word in enumerate(line):
            if word == 'B':
                if max_sc < max(j+1,len(line)-j+1):
                    max_sc = max(j+1,len(line)-j+1)
    for i,line in enumerate(train_Y):
        for j,word in enumerate(line):
            if word == 'B':
                if max_sc < max(j+1,len(line)-j+1):
                    max_sc = max(j+1,len(line)-j+1)
    return max_sc


def get_esc_data(): # evaluation sentiment classification
    SC = 4 #0,1,2,3
    list_test = get_basic_data(sentences_test_X,sentences_test_Y,'./data/test')
    list_train = get_basic_data(sentences_train_X,sentences_train_Y,'./data/train')
    B_SIZE_TRAIN = return_b_size(sentences_train_Y) #3329
    B_SIZE_TEST = return_b_size(sentences_test_Y) #364
    #MAX_SC = return_max_sc(sentences_test_Y,sentences_train_Y) #73 maybe no use.
    train_X_F = np.zeros((B_SIZE_TRAIN, MAX_LEN, 2 * WORD_DIM))
    train_X_B = np.zeros((B_SIZE_TRAIN, MAX_LEN, 2 * WORD_DIM))
    train_Y = np.zeros((B_SIZE_TRAIN, SC))
    test_X_F = np.zeros((B_SIZE_TEST, MAX_LEN, 2 * WORD_DIM))
    test_X_B = np.zeros((B_SIZE_TEST, MAX_LEN, 2 * WORD_DIM))
    test_Y = np.zeros((B_SIZE_TEST, SC))
    count = -1

    for i,line in enumerate(sentences_train_Y):
        for j,word in enumerate(line):
            if word == 'B':
                count += 1
                # get evaluation object vector
                i_count = 0
                while j+1 < len(line) and sentences_train_Y[i][j+i_count+1] == 'I':
                    i_count += 1
                temp_v = w2cmodel[sentences_train_X[i][j]]
                p_i_count = i_count
                while i_count > 0:
                    temp_v += w2cmodel[sentences_train_X[i][j+i_count]]
                    i_count -= 1
                if p_i_count > 0:
                    temp_v /= (p_i_count+1)

                # get this F and B data
                temp_data = np.zeros(WORD_DIM*2)
                for k , word in enumerate(sentences_train_X[i]):
                    if k < j :
                        train_X_F[count][k][:50] = w2cmodel[sentences_train_X[i][k]]
                        train_X_F[count][k][50:] = temp_v
                    if k == j or p_i_count > 0:
                        train_X_F[count][k][:50] = w2cmodel[sentences_train_X[i][k]]
                        train_X_F[count][k][50:] = temp_v
                        train_X_B[count][k-j][:50] = w2cmodel[sentences_train_X[i][k]]
                        train_X_B[count][k-j][50:] = temp_v
                        if k > j:
                            p_i_count -= 1
                    if k > j:
                        train_X_B[count][k-j][:50] = w2cmodel[sentences_train_X[i][k]]
                        train_X_B[count][k-j][50:] = temp_v
    count = -1
    for i, line in enumerate(sentences_test_Y):
        for j, word in enumerate(line):
            if word == 'B':
                count += 1
                # get evaluation object vector
                i_count = 0
                while j + 1 < len(line) and sentences_test_Y[i][j + i_count + 1] == 'I':
                    i_count += 1
                temp_v = w2cmodel[sentences_test_X[i][j]]
                p_i_count = i_count
                while i_count > 0:
                    temp_v += w2cmodel[sentences_test_X[i][j + i_count]]
                    i_count -= 1
                if p_i_count > 0:
                    temp_v /= (p_i_count + 1)

                # get this F and B data
                temp_data = np.zeros(WORD_DIM * 2)
                for k, word in enumerate(sentences_test_X[i]):
                    if k < j:
                        test_X_F[count][k][:50] = w2cmodel[sentences_test_X[i][k]]
                        test_X_F[count][k][50:] = temp_v
                    if k == j or p_i_count > 0:
                        test_X_F[count][k][:50] = w2cmodel[sentences_test_X[i][k]]
                        test_X_F[count][k][50:] = temp_v
                        test_X_B[count][k - j][:50] = w2cmodel[sentences_test_X[i][k]]
                        test_X_B[count][k - j][50:] = temp_v
                        if k > j:
                            p_i_count -= 1
                    if k > j:
                        test_X_B[count][k - j][:50] = w2cmodel[sentences_test_X[i][k]]
                        test_X_B[count][k - j][50:] = temp_v

    for i,line in enumerate(train_Y):
        line[list_train[i]] = 1
    for i,line in enumerate(test_Y):
        line[list_test[i]] = 1

    return train_X_F,train_X_B,train_Y,test_X_F,test_X_B,test_Y

if __name__ == "__main__":
    get_esc_data()