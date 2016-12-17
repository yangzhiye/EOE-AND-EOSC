from keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import get_data
MAX_LEN = 79
WORD_DIM = 50

def eoa_eva(test_Y,predict_test_Y):
    total = 0
    right = 0
    total_O = 0
    right_O = 0
    total_B = 0
    right_B = 0
    total_I = 0
    right_I = 0
    for i in range(test_Y.shape[0]):
        for j in range(MAX_LEN):
            if test_Y[i][j][0] == 1:
                continue
            total += 1
            if test_Y[i][j][1] == 1:
                total_O += 1
            if test_Y[i][j][2] == 1:
                total_B += 1
            if test_Y[i][j][3] == 1:
                total_I += 1
            if test_Y[i][j][predict_test_Y[i][j]] == 1:
                right += 1
                if test_Y[i][j][1] == 1:
                    right_O += 1
                if test_Y[i][j][2] == 1:
                    right_B += 1
                if test_Y[i][j][3] == 1:
                    right_I += 1

    print "total acc is %f" % (right * 1.0 / total)
    print "O acc is %f" % (right_O * 1.0 / total_O)
    print "B acc is %f" % (right_B * 1.0 / total_B)
    print "I acc is %f" % (right_I * 1.0 / total_I)

def eoa_eva1(test_Y,predict_test_Y): #(304,79,4),(304,79) {"X":0,"O":1,"B":2,"I":3}
    total = 0
    right = 0
    for i in range(test_Y.shape[0]):
        for j in range(MAX_LEN):
            if test_Y[i][j][0] == 1:
                break
            if test_Y[i][j][2] == 1:
                total+=1
                if predict_test_Y[i][j] == 2 and j == MAX_LEN - 1:
                    right+=1
                    continue
                if predict_test_Y[i][j] == 2 and predict_test_Y[i][j+1] == 1 and test_Y[i][j+1][1] == 1:
                    right+=1
                    continue
                if predict_test_Y[i][j] == 2 and test_Y[i][j+1][3] == 1:
                    flag = 1
                    while j < MAX_LEN - 1 and test_Y[i][j+1][3] == 1:
                        if predict_test_Y[i][j+1] != 3:
                            flag = 0
                            break
                        j+=1
                    if(flag == 1):
                        right+=1

    return right*1.0/total


def test_eoa(modelpath):
    train_X,train_Y,test_X,test_Y = get_data.return_eoa_data()
    lstmmodel = load_model(modelpath)
    predict_test_Y = lstmmodel.predict_classes(test_X)
    #eoa_eva(test_Y, predict_test_Y)
    return eoa_eva1(test_Y, predict_test_Y)


def test_eosc(modelpath):
    train_X_F, train_X_B, train_Y, test_X_F, test_X_B, test_Y = get_data.get_eosc_data()
    model = load_model(modelpath)
    predict_test_Y = model.predict_classes([test_X_F,test_X_B])
    print predict_test_Y
    ans = 0
    for i,n in enumerate(predict_test_Y):
        if test_Y[i][n] == 1:
            ans += 1
    return ans*1.0/len(predict_test_Y)


def test_a_sentence(eoe_model_path,eosc_model_path,w2c_model_path):
    #content = raw_input("input:")
    content = "the pizza is not good"
    content_list = content.split(" ")
    eoe_model = load_model(eoe_model_path)
    eosc_model = load_model(eosc_model_path)
    w2cmodel = Word2Vec.load(w2c_model_path)
    x = np.zeros((1,MAX_LEN,WORD_DIM))
    for i,word in enumerate(content_list):
        word = word.strip().lower()
        x[0][i] = w2cmodel[word]
    eoe_y = eoe_model.predict_classes(x)
    B_list = []
    dic_B = {}
    count = 0
    for i,result in enumerate(eoe_y[0]):
        if result == 2:
            temp_list = []
            temp_list.append(content_list[i])
            temp_i = i
            n = 0
            while temp_i + 1 < len(content_list) and eoe_y[0][temp_i+1] == 3:
                temp_list.append(content_list[temp_i+1])
                temp_i += 1
                n += 1
            dic_B[count] = [i,n]
            count += 1
            B_list.append(temp_list)
    print B_list
    x_f = np.zeros((len(B_list),MAX_LEN,2*WORD_DIM))
    x_b = np.zeros((len(B_list),MAX_LEN,2*WORD_DIM))
    B_vector = np.zeros((50))
    for i,line in enumerate(B_list):
        I_count = dic_B[i][1]
        pos = dic_B[i][0]
        for j,word in enumerate(line):
            word = word.strip().lower()
            B_vector += w2cmodel[word]
        B_vector /= len(line)
        for k, word in enumerate(content_list):
            word = word.strip().lower()
            if k < pos:
                x_f[i][k][:50] = w2cmodel[word]
                x_f[i][k][50:] = B_vector
            if k == j or I_count > 0:
                x_f[i][k][:50] = w2cmodel[word]
                x_f[i][k][50:] = B_vector
                x_b[i][k - j][:50] = w2cmodel[word]
                x_b[i][k - j][50:] = B_vector
                I_count -= 1
            if k > j:
                x_b[i][k - j][:50] = w2cmodel[content_list[k]]
                x_b[i][k - j][50:] = B_vector
    eosc_y = eosc_model.predict_classes([x_f, x_b])
    print eosc_y


if __name__ == "__main__":
    #print "eoe_lstm_model accuracy is %f" % test_eoa("./model/eoe_lstm_model")
    #print "eoe_GRU_model accuracy is %f" % test_eoa("./model/eoe_GRU_model")
    #print "eoe_Blstm_model accuracy is %f" % test_eoa("./model/eoe_Blstm_model")
    #print "eosc_lstm_model accuracy is %f" % test_eosc("./model/eosc_lstm_model")
    print "eosc_Blstm_model accuracy is %f" % test_eosc("./model/eosc_Blstm_model")
    #test_a_sentence("./model/eoe_lstm_model", "./model/eosc_Blstm_model", "./model/50features_1minwords_10context")




