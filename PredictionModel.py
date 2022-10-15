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
from tensorflow.keras.layers import Dense
from Credentials import credentials

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
data = pandas.DataFrame(cur.fetchall(), columns=['date', 'close'])
print(data)

# Plotting the SQL database to a graph.
plt.plot(data["date"], data["close"], label="BTC Price")
plt.title("BTC-USD Price")
plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.legend()
plt.show()

# Formatting data to prepare it for prediction.
X = data["close"].values.reshape(-1,1)
#y = data["date"].values.astype(pandas.Float32Dtype)
y = data["date"].values.astype(pandas.DatetimeIndex)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
X_train.shape
X_test.shape
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_train)

# Testing pulling index values.
print(data['date'].astype(str))

# Creating the model for prediction.
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.compile(
    optimizer = 'rmsprop', 
    loss = 'mse'
)
model.fit(x = X_train, y = y_train, epochs = 250)
