import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
import matplotlib

st.write("""
# Simple Stock Price App
Shown are the stock closing price and volume of Google!
""")

today1 = date.today()
today0= today1 - timedelta(days = 2) #yesterday
now= today0.strftime("%Y-%m-%d")
# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
tickerSymbols = ["MSFT", "GOOGL", "AAPL"]
#get data on this ticker
#tickerData = yf.tickerSymbols
data = yf.download(tickerSymbols, start='2010-01-20', end=now)
#print(tickerData)
#get the historical prices for this ticker
# tickers =['FNF', 'ASML', 'GOOGL', 'CVS']
# data = yf.download(tickers, group_by="ticker", period='1y')
#tickerDf = yf.download(tickerData, period='1y') #, start='2020-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits
#print(tickerDf)

st.write("""
## Closing Price
""")
st.line_chart(data.Close)
st.write("""
## Volume Price
""")
st.line_chart(data.Volume)

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)