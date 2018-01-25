import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
wo_test = pd.read_csv('without_test_thomas.csv', engine='python')
wo_train = pd.read_csv('without_train_thomas.csv', engine='python')
w_test = pd.read_csv('with_test_thomas.csv', engine='python')
w_train = pd.read_csv('with_train_thomas.csv', engine='python')
bitcoin = pd.read_csv('./data/combined.csv', usecols=[5], engine='python')

# plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], np.array(wo_test)[:,5], label="without Google Trends")
# plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], np.array(w_test)[:,5], label="with Google Trends")
# plt.legend(loc='best')
# plt.title('Without Google Trends - Train Set')
# plt.ylabel('Loss')
# plt.xlabel('Lookback (Hours)')
# plt.show()


##plot bitcoin, google trends)
df = pd.read_csv('./data/combined.csv', usecols=[2,3,4,5,6], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
print(df)
plt.plot(df[:,0], linewidth=0.2, label="Bitcoin Closing Price")
plt.plot(df[:,4], linewidth=0.2, label="Google Queries")
plt.title('Bitcoin price and Google Search Data')
plt.ylabel('Normalized Data')
plt.xlabel('Time (Hours)')
plt.legend(loc='best')
plt.show()