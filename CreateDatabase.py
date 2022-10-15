import yfinance
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from datetime import datetime, date
import numpy
from Credentials import credentials
from Functions import TradingFunctions
psycopg2.extensions.register_adapter(numpy.int64, psycopg2._psycopg.AsIs)

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
conn.autocommit = True
cur = conn.cursor()

# Define variables
start_date = datetime(2000, 1, 1)
end_date = datetime.now()
symbol = "BTC-USD"

# Pulling and reformatting data from Yahoo Finance.
# Pulls daily information from the past 20 years.
data = yfinance.download(tickers=symbol, period = '20y', interval = '1d')
data = data.rename(columns={"Adj Close": "Adj_Close"})
data['Ticker'] = symbol
records = data.to_records(index=True)

# Creating the table.
cur.execute('''CREATE TABLE IF NOT EXISTS daily_sma
               (
                Date DATE NOT NULL,
                Open FLOAT NOT NULL,
                High FLOAT NOT NULL,
                Low FLOAT NOT NULL,
                Close FLOAT NOT NULL,
                Adj_Close FLOAT NOT NULL,
                Volume BIGINT NOT NULL,
                Ticker VARCHAR(255) NOT NULL
                );''')  
print("Table created successfully.")

# Iterating over rows to handle duplicate entries.
for row in records:
    try:
        query = """INSERT INTO daily_sma (Date, Open, High, Low, Close, Adj_Close, Volume, Ticker)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"""
        cur = conn.cursor()
        cur.execute(query, row)
        print("Data Insert Successfully.")
    except:
        print("Row exists, skipping.")

# Set the date column as a unique key.
try:
    cur.execute("ALTER TABLE daily_sma ADD CONSTRAINT date_unique UNIQUE (date);")
    print("Date column set as unique key.")
except:
    print("The column date is already set as an unique key.")

# Closing the connection to the database.
conn.close()
