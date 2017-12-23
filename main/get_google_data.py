from pytrends.request import TrendReq
import datetime
import time
import pandas as pd

kw_list = ["bitcoin"]
start = datetime.datetime.strptime("2015-01-01","%Y-%M-%S")
delta= datetime.timedelta(days=7)
end = start+delta
number_of_weeks = 154
results = []

for t in range(number_of_weeks):
    end = start+delta
    time_input= str(start.year)+"-"+str(start.month).zfill(2) +"-"+str(start.day).zfill(2) +"T00 "+str(end.year)+"-"+str(end.month).zfill(2) +"-"+str(end.day).zfill(2) +"T00"
    print(time_input)
    pytrends = TrendReq(hl='en-US')
    print("requesting data for week: " + str(t))
    pytrends.build_payload(kw_list, cat=0, timeframe=time_input, geo='', gprop='')
    if t == 0: 
        corrected=pytrends.interest_over_time()
        print(corrected)
    else:
        new_week = pytrends.interest_over_time()
        old_val=corrected.iloc[-1]['bitcoin']
        new_val=new_week.iloc[1]['bitcoin']
        corr_val=float(old_val)/ float(new_val)
        corrected.loc[:, 'bitcoin'] /= corr_val
        frames=[corrected, new_week]
        print(new_week)
        corrected=pd.concat(frames)
        print(corrected)
    start=end
    time.sleep(0.5)

corrected.drop(['isPartial'], axis = 1, inplace = True)
corrected.to_csv("./data/google_trends.csv")
