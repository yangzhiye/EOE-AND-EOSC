import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Activation,Masking
from keras.layers import Conv1D, MaxPooling1D, Embedding,TimeDistributed
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import get_data
import sys
MAX_LEN = 79
WORD_DIM = 50

if __name__ == "__main__":
    train_X,train_Y,test_X,test_Y = get_data.return_data()
    print train_X.shape,train_Y.shape,test_X.shape,test_Y.shape
    dp = Sequential()
    #dp.add(Masking(mask_value=0.,input_shape=(MAX_LEN,WORD_DIM)))
    dp.add(LSTM(100,input_shape=(MAX_LEN,WORD_DIM),return_sequences=True))
    dp.add(Dropout(0.5))
    dp.add(LSTM(100,return_sequences=True))
    dp.add(Dropout(0.5))
    dp.add(TimeDistributed(Dense(4,activation="softmax")))
    dp.summary()
    dp.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',patience=2)
    dp.fit(train_X,train_Y,batch_size=40,validation_split=0.1,callbacks=[early_stopping])
    dp.save("./model/lstm_model")
