from keras.models import load_model
import numpy as np
import get_data
MAX_LEN = 79


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

    print "eva1 is %f" % (right*1.0/total)


def test_eoa():
    train_X,train_Y,test_X,test_Y = get_data.return_eoa_data()
    lstmmodel = load_model("./model/lstm_model")
    #print lstmmodel.metrics_names
    #print lstmmodel.evaluate(test_X,test_Y)
    predict_test_Y = lstmmodel.predict_classes(test_X)
    eoa_eva(test_Y, predict_test_Y)
    eoa_eva1(test_Y, predict_test_Y)


if __name__ == "__main__":
    test_eoa()




