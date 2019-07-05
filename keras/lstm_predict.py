import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation
import tushare as ts
from keras.callbacks import TensorBoard

df = ts.get_hist_data('000002',start='2017-12-12',end ='2019-05-01')
dd=df[['open','high','low','close']]
sortData = dd.sort_index()#按时间正序
frame = pd.DataFrame(sortData['close'])
scaler=MinMaxScaler()
def load_data(df,sequence_length=10,split=0.8):
    data_all = np.array(df).astype(float)
    data_all=scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i+sequence_length+1])
    reshaped_data=np.array(data).astype('float')
    x=reshaped_data[:,:-1]#对x进行归一化
    y=reshaped_data[:,-1]
    split_boundary=int(reshaped_data.shape[0]*split)
    train_x = x[:split_boundary]
    test_x = x[split_boundary:]
    train_y = y[:split_boundary]
    test_y = y[split_boundary:]
    return train_x,train_y,test_x,test_y

def build_model():
    model = Sequential()
    model.add(LSTM(input_dim = 1,output_dim = 6,return_sequences=True))
    model.add(LSTM(6,input_shape=(10,1),return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()
    return model

def train_model(train_x,train_y,test_x,test_y):
    model = build_model()
    try:
        callbacks = [TensorBoard(log_dir='/Users/hecy/opensource/ML/logDir',histogram_freq = 1)]
        model.fit(train_x,train_y,batch_size=512,nb_epoch=300,validation_split=0.1,callbacks=callbacks)
        predict = model.predict(test_x)
        predict = np.reshape(predict,(predict.size,))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    #print(predict)
    #print(test_y)
    try:
        fig=plt.figure(1)
        plt.plot(predict,'r:')
        plt.plot(test_y,'g-')
        plt.legend(['predict','true'])
    except Exception as e:
        print(e)
    return predict,test_y

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = load_data(frame, sequence_length=10, split=0.8)
    train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
    test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))
    predict_y,test_y = train_model(train_x,train_y,test_x,test_y)
    predict_y = scaler.inverse_transform([[i] for i in predict_y])
    test_y = scaler.inverse_transform(test_y)
    fig2=plt.figure(2)
    plt.plot(predict_y,'g:')
    plt.plot(test_y,'r-')
    plt.show()
