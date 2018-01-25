########################################
#ALWAYS UPDATE THE MODEL SAVE STATEMENT#
#needs: python -m pip install h5py     #
########################################

#delete to use GPU if applicable.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forward=1440):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-look_forward-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back + look_forward, 3])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

##zet deze naar 10
end = 20

#Look forward in hours.
time_stamps=[1,12,24,48,96,192,384]
ep=1000 #epochs
ba=250 #batchsize

##arrays for opslaan RMSE
with_train=numpy.zeros((20,len(time_stamps)))
without_train=numpy.zeros((20,len(time_stamps)))
with_test=numpy.zeros((20,len(time_stamps)))
without_test=numpy.zeros((20,len(time_stamps)))

models=[]
LB=10
LF=96

print(len(models))
# load the dataset
dataframe = read_csv('./data/combined.csv', usecols=[2, 3, 4, 5, 6], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float64')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+look_forward
# which is the number of previous time steps to use as input variables to predict the next time period
# number of hourse to use (look_back) to predict the future (look_forward)

print(LB,LF)
look_back = LB
look_forward = LF
trainX, trainY = create_dataset(train, look_back, look_forward)
testX, testY = create_dataset(test, look_back, look_forward)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))

##early stopping, once delta increases with some patience to prevent local optima
es = callbacks.EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=2,
                             verbose=2, mode='auto')

# create LSTM network, we can add more layers here.

models.append(Sequential())
model=models[-1]
model.add(LSTM(6, input_shape=(5, look_back), activation='tanh'))
model.add(Dense(1, activation='tanh'))

#define optimizer
sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

#optimize using mean_squared_error as loss fucntion.
model.compile(loss='mean_squared_error', optimizer='sgd')
history_w = model.fit(trainX, trainY, epochs=ep, batch_size=ba, verbose=2, shuffle=False, validation_split=0.2, callbacks=[es])

model.reset_states()

dataframe = read_csv('./data/combined.csv', usecols=[2, 3, 4, 5], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float64')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+look_forward
# which is the number of previous time steps to use as input variables to predict the next time period
# number of hourse to use (look_back) to predict the future (look_forward)
trainX, trainY = create_dataset(train, look_back, look_forward)
testX, testY = create_dataset(test, look_back, look_forward)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))

##early stopping, once delta increases with some patience to prevent local optima
es = callbacks.EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=2,
                             verbose=2, mode='auto')

# create LSTM network, we can add more layers here.
models.append(Sequential())
model=models[-1]
model.add(LSTM(6, input_shape=(4, look_back), activation='tanh'))
model.add(Dense(1, activation='tanh'))

# define optimizer
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# optimize using mean_squared_error as loss fucntion.
model.compile(loss='mean_squared_error', optimizer='sgd')
history_wo = model.fit(trainX, trainY, epochs=ep, batch_size=ba, verbose=2, shuffle=False, validation_split=0.2,
                    callbacks=[es])



# all=[history_wo.history['loss'][0:],history_wo.history['val_loss'][0:],history_w.history['loss'][0:],history_w.history['val_loss'][0:]]
# all=numpy.array(all)
# print(all)
# numpy.savetxt("loss_10_96_thomas.csv", all, delimiter=",")
##plot the validation error late.
plt.plot(history_wo.history['loss'][10:])
plt.plot(history_wo.history['val_loss'][10:])
plt.plot(history_w.history['loss'][10:])
plt.plot(history_w.history['val_loss'][10:])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train - wo', 'validation - wo', 'train - w', 'validation -w'], loc='best')
plt.show()

model.reset_states()