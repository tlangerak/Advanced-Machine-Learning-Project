import pandas as pd
from collections import OrderedDict


print("importing data...")
df = pd.read_csv('./data/btc_1min.csv', usecols=[1,2,3,4, 5], engine='python')
df["Timestamp"]=pd.to_datetime(df["Timestamp"], unit='s')


df.drop(df.index[[0,1,2,3,4,5,6,7]], inplace=True)
df = df.reset_index(drop=True)


all=[]
op, hg, lw, cl, dt=0, 0, 0, 0, 0
print("converting to hours, have some patience and get a coffee...")
for index, row in df.iterrows():
    if index == 0:
        op=row["Open"]
        hg=row["High"]
        lw=row["Low"]
        cl=row["Close"]
        dt = row["Timestamp"]

    else:
        if row["Timestamp"].minute == 0:
            op = row["Open"]
            hg = row["High"]
            lw = row["Low"]
            cl = row["Close"]
            dt = row["Timestamp"]

        if row["High"] > hg:
            hg = row["High"]

        if row["Low"] < lw:
            lw = row["Low"]

        if row["Timestamp"].minute == 59:
            cl = row["Close"]
            _hour = OrderedDict({'date':dt, 'Open':op, 'High': hg, 'Low': lw, 'Close': cl})
            all.append(_hour)
            op=0
            hg=0
            lw=0
            cl=0

df = pd.DataFrame(all)
df.to_csv('./data/btc_hourly.csv')

# gt = pd.read_csv('./data/google_trends.csv', engine='python')
# gt['bitcoin']=gt['bitcoin'].astype(float)
# print(gt)
# df['Open']=df['Open'].astype(float)
# df['High']=df['High'].astype(float)
# df['Low']=df['Low'].astype(float)
# df['Close']=df['Close'].astype(float)
# gt['bitcoin']=gt['bitcoin'].astype(str)
# print("combining data...")
# d_final=pd.merge(df, gt, on="date", how="inner")
# d_final.to_csv("./data/combined.csv")
# print(d_final)