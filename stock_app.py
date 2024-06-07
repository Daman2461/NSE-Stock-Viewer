import streamlit as st
import pandas as pd
import appdirs as ad
import yfinance as yf
import plotly.graph_objs as go

# Set custom cache directory for yfinance
ad.user_cache_dir = lambda *args: "/tmp"

# Set page layout to wide
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

# Define valid intervals for each period
valid_intervals = {
    '1d': ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'],
    '5d': ['5m', '15m', '30m', '60m', '90m', '1h'],
    '1mo': ['1h', '1d'],
    '3mo': ['1d', '5d'],
    '6mo': ['1d', '5d', '1wk'],
    '1y': ['1d', '5d', '1wk', '1mo'],
    '2y': ['1d', '5d', '1wk', '1mo'],
    '5y': ['1d', '5d', '1wk', '1mo', '3mo'],
    '10y': ['1d', '5d', '1wk', '1mo', '3mo'],
    'ytd': ['1d', '5d', '1wk', '1mo'],
    'max': ['1d', '5d', '1wk', '1mo', '3mo']
}

st.title('Stock App')

# User input for company
company = st.selectbox(
    "Company",
    ("Nifty 50", "Reliance Industries", "Tata Consultancy Services (TCS)", "HDFC Bank", "ICICI Bank", "Tata Steel", "Infosys"),
    index=0,
    key='company'
)

# Use radio buttons for Period selection
period = st.radio(
    "Period",
    options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
    index=2,
    key='period'
)

# Use radio buttons for Interval selection
interval = st.radio(
    "Interval",
    options=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
    index=8,
    key='interval'
)

# Validate interval based on the selected period
if interval not in valid_intervals[period]:
    st.warning(f"The selected interval '{interval}' is not valid for the period '{period}'. Setting to the maximum possible interval.")
    interval = valid_intervals[period][-1]

# Fetch historical data for the selected company
data = {}
stock_data = yf.download(tickers[company], period=period, interval=interval, progress=False)[['Open', 'Close']].tail(600)
data[company] = stock_data

# Convert the data to a DataFrame for better visualization
df = pd.concat(data, axis=1)
df.columns = pd.MultiIndex.from_product([[company], ['Open', 'Close']], names=['Company', 'Attributes'])

# Plot the selected company data with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[company]['Open'], mode='lines', name='Opening Prices'))
fig.add_trace(go.Scatter(x=df.index, y=df[company]['Close'], mode='lines', name='Closing Prices'))
fig.update_layout(
    title=f'{company} Stock Prices',
    xaxis_title='Date',
    yaxis_title='Price',
    width=800,
    height=500,
    xaxis_showgrid=True,
    yaxis_showgrid=True,
    hovermode='closest',
    xaxis={'showspikes': True}
)

# Display the graph and the dataframe in the same row
col1, col2 = st.columns([0.3, 0.7])
with col1:
    st.dataframe(df[company], use_container_width=True)
with col2:
    st.plotly_chart(fig)

st.write("Made by Daman")
