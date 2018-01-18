import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

wo_test = pd.read_csv('without_test.csv', engine='python')
wo_train = pd.read_csv('without_train.csv', engine='python')
w_test = pd.read_csv('with_test.csv', engine='python')
w_train = pd.read_csv('with_train.csv', engine='python')
bitcoin = pd.read_csv('./data/combined.csv', usecols=[5], engine='python')

[a,b,c,d,e,f,g]=plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], wo_test)
plt.legend([a,b,c,d,e,f,g],[1,12,24,48,96,192,384],loc='best')
plt.show()


##plot bitcoin, google trends)
# df = pd.read_csv('./data/combined.csv', usecols=[2,3,4,5,6], engine='python')
# scaler = MinMaxScaler(feature_range=(0, 1))
# df = scaler.fit_transform(df)
# print(df)
# plt.plot(df[:,0], linewidth=0.2, label="Bitcoin Closing Price")
# plt.plot(df[:,4], linewidth=0.2, label="Google Queries")
# plt.legend(loc='best')
# plt.show()