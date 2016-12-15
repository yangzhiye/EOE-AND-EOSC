import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Activation,Masking,Merge,Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding,TimeDistributed,AveragePooling1D
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU,Bidirectional
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.utils import np_utils
import get_data
import sys
MAX_LEN = 79
WORD_DIM = 50

def train_eoa_model_of_lstm():
    train_X,train_Y,test_X,test_Y = get_data.return_eoa_data()
    print train_X.shape,train_Y.shape,test_X.shape,test_Y.shape
    mymodel = Sequential()
    #dp.add(Masking(mask_value=0.,input_shape=(MAX_LEN,WORD_DIM)))
    mymodel.add(LSTM(100,input_shape=(MAX_LEN,WORD_DIM),return_sequences=True))
    mymodel.add(Dropout(0.5))
    mymodel.add(LSTM(100,return_sequences=True))
    mymodel.add(Dropout(0.5))
    mymodel.add(TimeDistributed(Dense(4,activation="softmax")))
    mymodel.summary()
    mymodel.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',patience=2)
    mymodel.fit(train_X,train_Y,batch_size=40,validation_split=0.1,callbacks=[early_stopping])
    mymodel.save("./model/eoa_lstm_model")


def train_eoa_model_of_GRU():
    train_X, train_Y, test_X, test_Y = get_data.return_eoa_data()
    print train_X.shape, train_Y.shape, test_X.shape, test_Y.shape
    mymodel = Sequential()
    # dp.add(Masking(mask_value=0.,input_shape=(MAX_LEN,WORD_DIM)))
    mymodel.add(GRU(100, input_shape=(MAX_LEN, WORD_DIM), return_sequences=True))
    mymodel.add(Dropout(0.5))
    mymodel.add(GRU(100, return_sequences=True))
    mymodel.add(Dropout(0.5))
    mymodel.add(TimeDistributed(Dense(4, activation="softmax")))
    mymodel.summary()
    mymodel.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    mymodel.fit(train_X, train_Y, batch_size=40, validation_split=0.1, callbacks=[early_stopping])
    mymodel.save("./model/eoa_GRU_model")


def train_eoa_model_of_Blstm():
    train_X, train_Y, test_X, test_Y = get_data.return_eoa_data()
    print train_X.shape, train_Y.shape, test_X.shape, test_Y.shape
    mymodel = Sequential()
    # dp.add(Masking(mask_value=0.,input_shape=(MAX_LEN,WORD_DIM)))
    mymodel.add(Bidirectional(LSTM(100, return_sequences=True),input_shape=(MAX_LEN, WORD_DIM)))
    mymodel.add(Dropout(0.5))
    mymodel.add(Bidirectional(LSTM(100, return_sequences=True)))
    mymodel.add(Dropout(0.5))
    mymodel.add(TimeDistributed(Dense(4, activation="softmax")))
    mymodel.summary()
    mymodel.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    mymodel.fit(train_X, train_Y, batch_size=40, validation_split=0.1, callbacks=[early_stopping])
    mymodel.save("./model/eoa_Blstm_model")


def train_eosc_model_of_lstm():
    train_X_F, train_X_B, train_Y, test_X_F, test_X_B, test_Y = get_data.get_eosc_data()
    #print train_X_F.shape, train_X_B.shape, train_Y.shape, test_X_F.shape, test_X_B.shape, test_Y.shape
    encoder_a = Sequential()
    encoder_a.add(Masking(mask_value=0., input_shape=(MAX_LEN, WORD_DIM*2)))
    encoder_a.add(LSTM(100))
    encoder_a.add(Dropout(0.5))

    encoder_b = Sequential()
    encoder_b.add(Masking(mask_value=0., input_shape=(MAX_LEN, WORD_DIM*2)))
    encoder_b.add(LSTM(100,go_backwards=True))
    encoder_b.add(Dropout(0.5))

    decoder = Sequential()
    decoder.add(Merge([encoder_a,encoder_b],mode='concat'))
    decoder.add(Dense(4,activation="softmax"))
    decoder.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    decoder.fit([train_X_F,train_X_B], train_Y, batch_size=40, validation_split=0.1, callbacks=[early_stopping])
    decoder.save("./model/eosc_lstm_model")


def train_eosc_model_of_Blstm():
    train_X_F, train_X_B, train_Y, test_X_F, test_X_B, test_Y = get_data.get_eosc_data()
    encoder_a = Sequential()
    #encoder_a.add(Masking(mask_value=0., input_shape=(MAX_LEN, WORD_DIM*2)))
    encoder_a.add(Bidirectional(LSTM(100,return_sequences=True),input_shape=(MAX_LEN, WORD_DIM*2)))
    encoder_a.add(Dropout(0.5))
    encoder_a.add(Lambda(lambda x:K.mean(x,axis=1),output_shape=lambda x:(x[0],x[2])))
    #encoder_a.summary()
    encoder_b = Sequential()
    #encoder_b.add(Masking(mask_value=0., input_shape=(MAX_LEN, WORD_DIM*2)))
    encoder_b.add(Bidirectional(LSTM(100,return_sequences=True,go_backwards=True),input_shape=(MAX_LEN, WORD_DIM*2)))
    encoder_b.add(Dropout(0.5))
    encoder_b.add(Lambda(lambda x:K.mean(x,axis=1),output_shape=lambda x:(x[0],x[2])))

    decoder = Sequential()
    decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
    decoder.add(Dense(4, activation="softmax"))
    #decoder.summary()
    decoder.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    decoder.fit([train_X_F, train_X_B], train_Y, batch_size=40, validation_split=0.1, callbacks=[early_stopping])
    decoder.save("./model/eosc_Blstm_model")




if __name__ == "__main__":
    #train_eoa_model_of_lstm()
    #train_eoa_model_of_GRU()
    #train_eoa_model_of_Blstm()
    #train_eosc_model_of_lstm()
    train_eosc_model_of_Blstm()