from keras.utils import plot_model
from keras.models import load_model
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

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

# load the dataset
dataframe = read_csv('./data/combined.csv', usecols=[2,3,4,5,6], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float64')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+look_forward
# which is the number of previous time steps to use as input variables to predict the next time period
# number of hourse to use (look_back) to predict the future (look_forward)
look_back = 10
look_forward = 1
trainX, trainY = create_dataset(train, look_back, look_forward)
testX, testY = create_dataset(test, look_back, look_forward)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))

model=load_model('with_googletrends_lstm6_dense1_epochs2000_batchsize250_validation021515416205.h5')

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:]))
print('Train Score: %.4f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:]))
print('Test Score: %.4f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+look_forward+1:len(dataset)-look_forward-1, :] = testPredict

# plot baseline and predictions
plt.plot(dataset[:,3], linewidth=0.5, label='dataset', color="blue")
plt.plot(trainPredictPlot, linewidth=0.5, label='predict on train', color="red")
plt.plot(testPredictPlot, linewidth=0.5, label='predict on test', color="green")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

plot_model(model, to_file='test.png', show_shapes=True)