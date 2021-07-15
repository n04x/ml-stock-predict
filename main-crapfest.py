from matplotlib import scale
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

data_source = 'alphavantage'

if data_source == 'alphavantage':
    # Loading data from Alpha Vantage
    api_key = 'YNSYV764TKT1EHPH'

    # AMD stock market prices
    ticker = 'AMD'

    # JSON file with all the stock market data from last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}".format(ticker,api_key)
    # Save data to file
    file_to_save = 'stock_market_data-{}.csv'.format(ticker)

    # if you haven't already saved the data,
    # go ahead and grab the data from the url
    # and store date, low, high, volume, close, open values to a Padas DataFrame
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['3. low']), float(v['2. high']), float(v['4. close']), float(v['1. open'])]
                df.loc[-1,:]= data_row
                df.index = df.index + 1
            
        print('Data saved to : {}'.format(file_to_save))
        df.to_csv(file_to_save)
    
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)

# Sort DataFrame by date
df = df.sort_values('Date')

# Double check the result
df.head()

# First calculate the mid prices from the highest and lowest
high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
mid_prices = (high_prices+low_prices)/2.0
print(len(mid_prices))
# Split the training data and test data
train_data = mid_prices[:2730]
test_data = mid_prices[2730:]

# Scale the data to be between 0 and 1
scaler = MinMaxScaler
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

# Train the Scaler with training data and smooth data
smoothing_window_size = 680

for di in range(0, 5000, smoothing_window_size):
    print(di)
    scaler.fit(train_data[di:di + smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# Normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# reshape both train and test data
train_data = train_data.reshape(-1)

# normalize test data
test_data = scaler.transform(test_data).reshape(-1)

train_data = train_data.reshape(-1,1)

# Exponential moving average smoothing 
# so the data will have smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1

for ti in range(11000):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

# Used for visualization and test purpose
all_mid_data = np.concatenate([train_data,test_data],axis=0)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):
    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx, 'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx - window_size: pred_idx]))
    mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: {:.5f}'.format(0.5*np.mean(mse_errors)))
    
# Data Visualization
plt.figure(figsize= (18,9))
plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)

plt.show()


