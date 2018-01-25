#delete to use GPU if applicable.
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.advanced_activations import LeakyReLU


# fix random seed for reproducibility
numpy.random.seed(7)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forward=1440):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-look_forward-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back + look_forward, 3])
	return numpy.array(dataX), numpy.array(dataY)

def create_model(inputs, n_layers, n_nodes):
    model = Sequential()
    if n_layers > 1:
        model.add(LSTM(n_nodes, return_sequences = True, input_shape=(inputs, look_back), activation='tanh'))
        for n in range(1,n_layers-1):
            model.add(LSTM(n_nodes, return_sequences = True, activation='tanh'))
        model.add(LSTM(n_nodes, activation='tanh'))
    else:
        model.add(LSTM(n_nodes, input_shape=(inputs, look_back), activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    return model

ep=5000
ba=250
look_back= 10
look_forward= 96
models=[]

##arrays for opslaan RMSE
with_train=numpy.zeros((10,5))
without_train=numpy.zeros((10,5))
with_test=numpy.zeros((10,5))
without_test=numpy.zeros((10,5))
x=0
y=0
for n in range(61,102,10):
    y=0
    for l in range(1,6):
        print("Number of Nodes: "+str(n)+"\t Number of Layers: "+ str(l))
        dataframe = read_csv('./data/combined.csv', usecols=[2, 3, 4, 5, 6], engine='python', skipfooter=3)
        dataset = dataframe.values
        dataset = dataset.astype('float64')

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
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

        models.append(create_model(5,l,n))
        model=models[-1]

        #optimize using mean_squared_error as loss fucntion.
        model.compile(loss='mean_squared_error', optimizer='sgd')
        history = model.fit(trainX, trainY, epochs=ep, batch_size=ba, verbose=2, shuffle=False, validation_split=0.2, callbacks=[es])
        model.save("multilayer_with_" +str(n) + "_" + str(l) + '.h5')

        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:]))
        print('Train Score: %.4f RMSE' % (trainScore))
        with_train[x,y]=trainScore
        testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:]))
        print('Test Score: %.4f RMSE' % (testScore))
        with_test[x, y] = testScore

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
        plt.ylabel('predict')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.savefig("multi_with_"+str(n)+"_"+str(l)+'_predict.png')
        plt.clf()

        ##plot the validation error late.
        plt.plot(history.history['loss'][0:])
        plt.plot(history.history['val_loss'][0:])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig("multi_with_"+str(n) + "_" + str(l) + '_error.png')
        plt.clf()

        model.reset_states()

        dataframe = read_csv('./data/combined.csv', usecols=[2, 3, 4, 5], engine='python', skipfooter=3)
        dataset = dataframe.values
        dataset = dataset.astype('float64')

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
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
        models.append(create_model(4, l, n))
        model = models[-1]

        # optimize using mean_squared_error as loss fucntion.
        model.compile(loss='mean_squared_error', optimizer='sgd')
        history = model.fit(trainX, trainY, epochs=ep, batch_size=ba, verbose=2, shuffle=False, validation_split=0.2,
                            callbacks=[es])
        model.save("multi_without_" +str(n) + "_" + str(l) + '.h5')
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:]))
        print('Train Score: %.4f RMSE' % (trainScore))
        without_train[x,y]=trainScore
        testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:]))
        print('Test Score: %.4f RMSE' % (testScore))
        without_test[x, y] = testScore

        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + look_forward + 1:len(dataset) - look_forward - 1,
        :] = testPredict

        # plot baseline and predictions
        plt.plot(dataset[:,3], linewidth=0.5, label='dataset', color="blue")
        plt.plot(trainPredictPlot, linewidth=0.5, label='predict on train', color="red")
        plt.plot(testPredictPlot, linewidth=0.5, label='predict on test', color="green")
        plt.ylabel('predicted')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.savefig("multi_without_"+str(n)+"_"+str(l)+'_predict.png')
        plt.clf()

        ##plot the validation error late.
        plt.plot(history.history['loss'][0:])
        plt.plot(history.history['val_loss'][0:])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig("multi_without_"+str(n) + "_" + str(l) + '_error.png')
        plt.clf()


        model.reset_states()

        numpy.savetxt("multi_without_test_thomas.csv", without_test, delimiter=",")
        numpy.savetxt("multi_with_test_thomas.csv", with_test, delimiter=",")
        numpy.savetxt("multi_without_train_thomas.csv", without_train, delimiter=",")
        numpy.savetxt("multi_with_train_thomas.csv", with_train, delimiter=",")
        y+=1
    x+=1