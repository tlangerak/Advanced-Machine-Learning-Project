import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./data/combined.csv', usecols=[5,6], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
print(df)
plt.plot(df[:,0], linewidth=0.2, label="Bitcoin Closing Price")
plt.plot(df[:,1], linewidth=0.2, label="Google Queries")
plt.legend(loc='best')
plt.show()