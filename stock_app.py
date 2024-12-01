import streamlit as st
import pandas as pd
import appdirs as ad
import random
import yfinance as yf
import plotly.graph_objs as go
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm

# Set up the cache directory
ad.user_cache_dir = lambda *args: "/tmp"

st.set_page_config(
    page_title="NSE Stock Viewer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

with st.sidebar:
    company = st.selectbox(
        "Company",
        ("Nifty 50", "Reliance Industries", "Tata Consultancy Services (TCS)", "HDFC Bank", "ICICI Bank", "Tata Steel", "Infosys"),
        index=3,
        key='company'
    )

    period = st.selectbox(
        "Period",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
        index=0,
        key='period'
    )

    # Filter valid intervals based on the selected period
    available_intervals = valid_intervals.get(period, [])
    
    interval = st.selectbox(
        "Interval",
        available_intervals,
        index=available_intervals.index('1d') if '1d' in available_intervals else 0,
        key='interval'
    )

if interval not in valid_intervals[period]:
    interval = valid_intervals[period][-1]

# Fetch historical OHLCV data for the selected company
data = {}
stock_data = yf.download(tickers[company], period=period, interval=interval, progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']].tail(600)
data[company] = stock_data

# Convert the data to a DataFrame for better visualization
df = pd.concat(data, axis=1)
df.columns = pd.MultiIndex.from_product([[company], ['Open', 'High', 'Low', 'Close', 'Volume']], names=['Company', 'Attributes'])

# Plot the selected company data with Plotly using candlestick chart
fig = go.Figure(
    data=[go.Candlestick(
        x=df.index,
        open=df[company]['Open'],
        high=df[company]['High'],
        low=df[company]['Low'],
        close=df[company]['Close'],
        name='OHLC'
    )]
)

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

# Display the candlestick chart and the dataframe in the same row
st.plotly_chart(fig)
st.dataframe(df[company], use_container_width=True)

# casc mod (on Close Price)
def casc_mod(y, x, frac=0.035):
    lowess = sm.nonparametric.lowess
    smoothed = lowess(y, x, frac=frac)
    return smoothed[:, 1]

# Extract the closing prices and convert dates to numerical format
y = df[company]['Close'].values
x = np.arange(len(df))  # Numerical representation of dates

# Apply casc mod
smoothed_y = casc_mod(y, x, frac=0.035)

# Plot casc smoothed line
fig_casc = go.Figure()

# Add the original closing prices
fig_casc.add_trace(go.Scatter(x=df.index, y=y, mode='lines', name="Closing Price", line=dict(color='blue')))

# Add the casc smoothed line
fig_casc.add_trace(go.Scatter(x=df.index, y=smoothed_y, mode='lines', name="Cascading Smoothed", line=dict(color='red', width=3)))

fig_casc.update_layout(
    title=f'{company} Closing Price with Cascading Model',
    xaxis_title='Date',
    yaxis_title='Price',
    width=800,
    height=500,
    xaxis_showgrid=True,
    yaxis_showgrid=True,
    hovermode='x'
)

st.plotly_chart(fig_casc)

# Simulated MARL-based Option Pricing  
def marl_based_option_pricing(S, K, T, r, sigma, option_type='call'):
    
    # Simulate MARL agent decision process
    st.write("The MARL agent is evaluating the option pricing...")
    action = random.choice(['buy', 'sell'])  # Simulated agent decision (buy or sell)
    
   
    option_price = agent(S, K, T, r, sigma, option_type)
    
    st.write(f"MARL Agent Decision: {action.capitalize()} the option")
    return option_price
 
def agent(S, K, T, r, sigma, option_type='call'):
     
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate call or put option price based on the option type
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

# Example of using MARL-based option pricing
S = float(df[company]['Close'].iloc[-1] ) # Latest stock price
K = st.number_input("Strike Price", value=S, step=1.0)  # User-input strike price
T = st.number_input("Time to Maturity (in years)", value=1.0)  # User-input time to maturity
r = 0.05
sigma = 0.2

# Option type selection
option_type = st.selectbox("Option Type", ["call", "put"])

 
option_price = marl_based_option_pricing(S, K, T, r, sigma, option_type)

st.write(f"MARL-based {option_type.capitalize()} option price: ₹{option_price:.2f}")
 
