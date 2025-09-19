import datetime
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
#libraries
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
#import tabulate as tabulate
import pandas as pd
from arch import arch_model
import requests
import warnings
#warnings.filterwarnings("always")
warnings.filterwarnings("ignore") #i think i have made the relevant guardrails for working with the data from the getters, but some
#FutureWarnings keep popping up. will KIV though,

# IV getter: Dolthub (the source) updates every other day. But since we use a composite of RV/Garch/IV, (with discretionary lookback, etc.
#i dont think it matters too much.)
def get_iv(ticker):
    url = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
    

    query = f"""
    SELECT iv_current
    FROM `volatility_history`
    WHERE act_symbol = '{ticker}'
    ORDER BY `date` DESC
    LIMIT 1;
    """

    params = {"q": query}
    try: 
        response = requests.get(url, params=params, timeout = 5)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        st.warning(f"Timeout: {ticker} IV request took too long")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Request error for {ticker}: {e}")
        return None
    except ValueError:
        st.warning(f"Invalid JSON returned for {ticker}")
        return None
    # Extract only the rows field
    rows = data.get("rows", [])
    
    # Convert rows into DataFrame
    df = pd.DataFrame(rows)
        #print(df)
    if not df.empty:
        ser = df["iv_current"] 
        return float(ser.iloc[0])
    else:
        return None

#Spot,rv,garch Getter (uses tha same yfin price series for all, save compute/fetch time)
#rv lookback: 20 trading days
#Garch lookback: 2 years.
def get_spot_rv_garch(ticker,duration):
    try:
        data = yf.download(ticker, period = "500d", interval="1d", auto_adjust=False)
    except Exception:
        return None, None, None
    if data.empty:
        return None,None,None
    ser = data["Close"].squeeze().dropna()
    ser = ser[ser>0]
    spot = float(ser.iat[-1])
    
    data["LogRet"] = np.log(data["Close"] / data["Close"].shift(1))
    returns = data["LogRet"].dropna()
    
    rv = np.std(returns[-20:], ddof=1) *np.sqrt(252)
    
    model = arch_model(returns * 100, vol="Garch", p=1, q=1)  # returns in %
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=252, reindex=False)

    var_forecast = forecast.variance.values[-1]

    daily_vol_forecast = np.sqrt(var_forecast) / 100
    # GARCH forecasted vol for the duration selected by user (annualized)
    garch = np.sqrt(np.sum(daily_vol_forecast[:duration]**2)) * np.sqrt(252/duration)
    
    return spot, rv, garch
st.title("Auto Volatility Position Sizer")

mode = st.radio("Choose Mode:" , ["Equities (fetch Spot, IV, RV, GARCH)", "Futures, FX","Manual (enter Spot, Vol, etc.)"] )
if mode == "Equities (fetch Spot, IV, RV, GARCH)":
    ticker = st.text_input("Ticker Symbol", "AAPL")
    amount = st.number_input("Amount you are willing to lose", min_value=100.0, step=100.0)
    duration = st.number_input("Duration (Trading days)", min_value=1, max_value=252, value=20)
    direction = st.radio("Direction", ["long", "short"])

    if st.button("Fetch Data"):
        st.session_state.iv = get_iv(ticker)
        st.session_state.spot, st.session_state.rv, st.session_state.garch = get_spot_rv_garch(ticker, duration)

    # Initialize session state if not set yet
    for key in ["spot", "iv", "rv", "garch"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Show fetched values + manual fallback inputs
    st.write("Fetched Variables (fill missing values manually if needed):")
    st.session_state.spot = st.number_input("Spot Price (yfinance)", value=st.session_state.spot if st.session_state.spot else 0.0,format="%.6f")
    st.session_state.iv = st.number_input("IV (annualised)(dolthub, updated every other day)", value=st.session_state.iv if st.session_state.iv else 0.0,format="%.6f")
    st.session_state.rv = st.number_input("RV (annualised)(yfinance)", value=st.session_state.rv if st.session_state.rv else 0.0,format="%.6f")
    st.session_state.garch = st.number_input("GARCH Vol (annualised)(yfinance)", value=st.session_state.garch if st.session_state.garch else 0.0,format="%.6f")

    # Calculation only after user clicks explicitly
    if st.button("Run Calculation"):
        spot = st.session_state.spot
        iv = st.session_state.iv
        rv = st.session_state.rv
        garch = st.session_state.garch

        if all(v > 0 for v in [spot, iv, rv, garch]):
            vol = (iv + rv + garch) / 3
            one_std = vol / np.sqrt(252) * np.sqrt(duration)
            one_half_std = one_std * 1.5

            if direction == "long":
                one_std_size = amount / (spot * one_std)
                one_std_stop = spot * (1 - one_std)
                one_half_std_size = amount / (spot * one_half_std)
                one_half_std_stop = spot * (1 - one_half_std)
            else:  # short
                one_std_size = amount / (spot * one_std)
                one_std_stop = spot * (1 + one_std)
                one_half_std_size = amount / (spot * one_half_std)
                one_half_std_stop = spot * (1 + one_half_std)

            df = pd.DataFrame([[direction, one_std_size, one_std_stop,
                                one_half_std_size, one_half_std_stop]],
                              columns=['Direction','1 SD Size','1 SD Stop Loss',
                                       '1.5 SD Size','1.5 SD Stop Loss'])
            st.table(df)
        else:
            st.error("Please ensure all values are filled in before calculation.")
elif mode == "Futures, FX":
    spot = st.number_input("Spot Price", min_value=0.000001,format="%.5f")
    iv = st.number_input("Implied Volatility (annualised)", min_value=0.0, step=0.01)
    amount = st.number_input("Amount you are willing to lose", min_value=100.0, step=100.0)
    
    # Separate pip size and pip value
    pip_size = st.number_input("Pip size (price movement per pip)", value=0.0001, format="%.5f")
    pip_value = st.number_input("Value per pip (in account currency)", min_value=0.01, step=0.01,format="%.5f")
    
    duration = st.number_input("Duration (days)", min_value=1, max_value=252, value=20)
    direction = st.radio("Direction", ["long", "short"])

    if st.button("Calculate Position"):
        # Scale volatility to time horizon
        one_std = iv / np.sqrt(252) * np.sqrt(duration)
        one_point_five_std = one_std * 1.5

        # Compute stop levels
        if direction == "long":
            one_std_stop = spot * (1 - one_std)
            one_point_five_std_stop = spot * (1 - one_point_five_std)
        else:  # short
            one_std_stop = spot * (1 + one_std)
            one_point_five_std_stop = spot * (1 + one_point_five_std)

        # Convert price difference into pip count
        price_diff_pips_1 = (spot - one_std_stop) / pip_size
        price_diff_pips_1_5 = (spot - one_point_five_std_stop) / pip_size

        # Position sizing based on monetary risk
        one_std_size = amount / (price_diff_pips_1 * pip_value)
        one_point_five_std_size = amount / (price_diff_pips_1_5 * pip_value)

        df = pd.DataFrame(
            [[direction, one_std_size, one_std_stop,
              one_point_five_std_size, one_point_five_std_stop]],
            columns=['Direction','1 SD Size','1 SD Stop Loss',
                     '1.5 SD Size','1.5 SD Stop Loss']
        )
        st.table(df)
elif mode == "Manual (enter Spot, Vol, etc.)":
    spot = st.number_input("Spot Price", min_value=0.01)
    iv = st.number_input("Implied Volatility (annualised)", min_value=0.0, step=0.01)
    amount = st.number_input("Amount you are willing to use", min_value=100.0, step=100.0)
    duration = st.number_input("Duration (days)", min_value=1, max_value=252, value=20)
    direction = st.radio("Direction", ["long", "short"])

    if st.button("Calculate Manual Position"):
        one_std = iv / np.sqrt(252) * np.sqrt(duration)
        one_half_std = one_std * 1.5

        if direction == "long":
            one_std_size = amount / (spot * one_std)
            one_std_stop = spot * (1 - one_std)
            one_half_std_size = amount / (spot * one_half_std)
            one_half_std_stop = spot * (1 - one_half_std)
        else:  # short
            one_std_size = amount / (spot * one_std)
            one_std_stop = spot * (1 + one_std)
            one_half_std_size = amount / (spot * one_half_std)
            one_half_std_stop = spot * (1 + one_half_std)

        df = pd.DataFrame([[direction, one_std_size, one_std_stop,
                            one_half_std_size, one_half_std_stop]],
                          columns=['Direction','1 SD Size','1 SD Stop Loss',
                                   '1.5 SD Size','1.5 SD Stop Loss'])
        st.table(df)