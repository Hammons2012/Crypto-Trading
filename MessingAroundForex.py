import yfinance
from datetime import datetime, date
import talib
import matplotlib.pyplot as plt

# Define variables
start_date = datetime(2020, 1, 1)
end_date = datetime.now()
symbol = "^NDX"

# Pulling the data.
data = yfinance.download(tickers=symbol, start = start_date, end = end_date, interval = '1d')
data = data.rename(columns={"Adj Close": "Adj_Close"})
data["SMA_5"] = talib.SMA(data["Close"],timeperiod=5)
data["MACD"], data["MACD_Sig"], data["MACD_Hist"] = talib.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
print(data)

# Plotting the yfinance data to a graph.
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle("NASDAQ 100 MACD Info - 1 Year each Day")
fig.set_size_inches(10,10)
ax1.plot(data["Close"], label = "NASDAQ 100 Closing Price")
ax1.legend()
ax2.plot(data["MACD"], label = "NASDAQ 100 MACD")
ax2.plot(data["MACD_Sig"], label = "NASDAQ 100 MACD Signal")
ax2.legend()
ax3.hist(data["MACD_Hist"], label = "NASDAQ 100 MACD Histogram")
ax3.legend()
plt.show()
