from cProfile import label
from pickletools import optimize
from datetime import datetime
import numpy
import pandas
import psycopg2
import matplotlib.pyplot as plt
import seaborn
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from Credentials import credentials

# Creating variables that will be used trhoughout the script.
test_size = 30 # in days
length = 25
n_features = 2

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

#Columns of price data to use
cur.execute("SELECT date, close FROM daily_sma;")
data = pandas.DataFrame(cur.fetchall(), columns = ['date', 'close'])
print(data)

# Plotting the SQL database to a graph.
plt.plot(data['date'], data['close'], label = "BTC Price")
plt.title("BTC-USD Price")
plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.legend()
plt.show()

# Formatting data to prepare it for prediction.
data['date'] = pandas.to_numeric(pandas.to_datetime(data['date']))
test_indicators = len(data) - test_size
train = data.iloc[:test_indicators]
test = data.iloc[test_indicators:]
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
generator = TimeseriesGenerator(scaled_train, scaled_train, length = length, batch_size = 1)
X, y = generator[0]

# Creating the model for prediction.
early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)
validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length = length, batch_size = 1)

model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape = (length, n_features)))
model.add(Dense(1))
model.compile(
    optimizer = 'adam',
    loss = 'mse'
)
model.summary()
model.fit(
    generator, 
    epochs = 20, 
    validation_data = validation_generator, 
    callbacks = early_stop
)

# Plotting losses.
losses = pandas.DataFrame(model.history.history)
losses.plot()
plt.show()

test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    test_predictions.append(current_batch)

true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.plot(figsize = (12, 8))
