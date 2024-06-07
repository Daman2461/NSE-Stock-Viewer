import streamlit as st
import pandas as pd

import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import yfinance as yf
import plotly.graph_objs as go
st.set_page_config(layout="wide")
# Define the ticker symbols for the selected Nifty 50 companies
tickers = {
    "Nifty 50": "^NSEI",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Steel": "TATASTEEL.NS"
}

st.title('Stock App')
with st.sidebar:
    company =  st.selectbox(
   "Company",
   ("Nifty 50","Reliance Industries","Tata Consultancy Services (TCS)", "HDFC Bank", "ICICI Bank", "Tata Steel","Infosys"),
   index=0,
   key='company'
    )

    period = st.selectbox(
   "Period",
   ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'),
   index=2,
   key='period'
    )
    interval = st.selectbox(
   "Interval",
   ('1m', '2m','5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'),
   index=8,
   key='interval'
    )

# Fetch historical data for the selected company
data = {}
stock_data = yf.download(tickers[company], period=period, interval=interval,progress=False)[['Open', 'Close']].tail(600)
data[company] = stock_data

# Convert the data to a DataFrame for better visualization
df = pd.concat(data, axis=1)
df.columns = pd.MultiIndex.from_product([[company], ['Open', 'Close']], names=['Company', 'Attributes'])

# Plot the selected company data with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[company]['Open'], mode='lines', name='Opening Prices'))
fig.update_layout( 
                  xaxis_title='Date',
                  yaxis_title='Price',
                  width=800, height=500)
fig.add_trace(go.Scatter(x=df.index, y=df[company]['Close'], mode='lines', name='Closing Prices'))
fig.update_layout(
    title=f'{company} Stock Prices',
    xaxis_title='Date',
    yaxis_title='Price',
    width=800,
    height=500,
    xaxis_showgrid=True,
    yaxis_showgrid=True,
    hovermode='x',
    xaxis={'showspikes': True}
)

# Display the graph and the dataframe in the same row

 
st.plotly_chart(fig)
st.dataframe(df[company],use_container_width=True)
 


st.write("Made by Daman")
