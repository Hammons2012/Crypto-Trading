import pandas as pd
import numpy as np
import math
import datetime as dt
import yfinance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from itertools import cycle
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from Credentials import credentials

# Creating variables that will be used trhoughout the script.
prediction_days = 30

# Creating connection to PG for psycopg2.
# Note that this script assumes that the database has already been created.
# sudo -u postgres psql
# CREATE DATABASE <database name>;
conn = psycopg2.connect(
    database = credentials.test_db_name, 
    user = credentials.test_db_user, 
    password = credentials.test_db_password, 
    host = '127.0.0.1', 
    port = '5432'
)

# Creating cursor for SQL query execution.
cur = conn.cursor()

# Columns of price data to use
cur.execute("SELECT date, close FROM daily_sma;")
data = pd.DataFrame(cur.fetchall(), columns=['date', 'close'])
data.shape
data_datetype = data.astype({'date': 'datetime64'})
data_datetype['date'] = pd.to_datetime(data_datetype['date'], unit = 's').dt.date
group = data_datetype.groupby('date')
closing_price_groupby_date = group['close'].mean()
prediction_days = 30

# Plotting the SQL database to a graph.
plt.plot(data['date'], data['close'], label = "BTC Price")
plt.title("BTC-USD Price")
plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.legend()
plt.show()

# Formatting the data in preparations for the predictions.
train = closing_price_groupby_date[:len(closing_price_groupby_date)-prediction_days].values.reshape(-1,1)
test = closing_price_groupby_date[len(closing_price_groupby_date)-prediction_days:].values.reshape(-1,1)
test.shape
chosen_col = 'Close'
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(train)
scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(test)

def dataset_generator_lstm(dataset, look_back = 1):
    # A “lookback period” defines the window-size of how many
    # previous timesteps are used in order to predict
    # the subsequent timestep. 
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0]) # this is the label or actual y-value
    return np.array(dataX), np.array(dataY)

trainX, trainY = dataset_generator_lstm(scaled_train)
testX, testY = dataset_generator_lstm(scaled_test)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1 ))

# Building the model.
model = Sequential()
model.add(LSTM(units = 100, activation = 'relu',return_sequences=True, input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 100, input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.summary()

# Compiling the LSTM
early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)
callbacks = [early_stop]
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(trainX, 
              trainY, 
              batch_size = 35, 
              epochs = 600, 
              verbose = 1, 
              shuffle = False, 
              validation_data = (testX, testY), 
              callbacks = callbacks)

# Plotting the model.
plt.figure(figsize=(16,7))
plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Remodeling the test data and plotting it.
predicted_test_data = model.predict(testX)
predicted_test_data = scaler_test.inverse_transform(predicted_test_data.reshape(-1, 1))
test_actual = scaler_test.inverse_transform(testY.reshape(-1, 1))

plt.figure(figsize=(16,7))
plt.plot(predicted_test_data, 'r', marker='.', label='Predicted Test')
plt.plot(test_actual, marker='.', label='Actual Test')
plt.legend()
plt.show()

# Remodeling the train data and plotting it.
predicted_train_data = model.predict(trainX)
predicted_train_data = scaler_train.inverse_transform(predicted_train_data.reshape(-1, 1))
train_actual = scaler_train.inverse_transform(trainY.reshape(-1, 1))

plt.figure(figsize=(16,7))
plt.plot(predicted_train_data, 'r', marker='.', label='Predicted Train')
plt.plot(train_actual, marker='.', label='Actual Train')
plt.legend()
plt.show()
rmse_lstm_test = math.sqrt(mean_squared_error(test_actual, predicted_test_data))

print('Test RMSE: %.3f' % rmse_lstm_test)

# Forecast for the amount of days in the lookback_period.
testX
testX.shape
lookback_period = 7
testX_last_loockback_days = testX[testX.shape[0] - lookback_period :  ]
testX_last_loockback_days.shape
testX_last_loockback_days
predicted_days_forecast_price_test = []

for i in range(lookback_period):  
  predicted_forecast_price_test_x = model.predict(testX_last_loockback_days[i:i+1])
  predicted_forecast_price_test_x = scaler_test.inverse_transform(predicted_forecast_price_test_x.reshape(-1, 1))
  predicted_days_forecast_price_test.append(predicted_forecast_price_test_x)

print(f"Forecast for the next {lookback_period} Days Beyond the actual trading days:", 
                                    np.array(predicted_days_forecast_price_test)) 

# That is the original Trading data ended on 30-Oct-2021, but now I am going to forecast beyond 30-Oct-2021
predicted_forecast_price_test_x = np.array(predicted_days_forecast_price_test)
predicted_forecast_price_test_x.shape
predicted_test_data.shape
predicted_test_data
predicted_forecast_price_test_x
predicted_forecast_price_test_x = predicted_forecast_price_test_x.flatten()
predicted_forecast_price_test_x
predicted_btc_price_test_data = predicted_test_data.flatten()
predicted_btc_price_test_data
predicted_btc_test_concatenated = np.concatenate((predicted_btc_price_test_data, predicted_forecast_price_test_x))
predicted_btc_test_concatenated
predicted_btc_test_concatenated.shape

plt.figure(figsize=(16,7))
plt.plot(predicted_btc_test_concatenated, 'r', marker='.', label='Predicted Test')
plt.plot(test_actual, marker='.', label='Actual Test')
plt.legend()
plt.show()
