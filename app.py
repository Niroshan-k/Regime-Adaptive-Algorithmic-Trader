import streamlit as st
import pandas as pd

st.set_page_config(page_title="RAAT Dashboard", layout="wide")

st.title("🦎 RAAT: Regime Adaptive Algorithmic Trader")
st.markdown("### Live AI Trading Signals")

# 1. Load Data
try:
    df = pd.read_csv('trade_log.csv')
    
    # 2. Key Metrics (The Top Bar)
    latest = df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latest Signal", latest['Final_Signal'])
    with col2:
        st.metric("Current Regime", f"Regime {latest['Regime']}")
    with col3:
        st.metric("Last Price", f"${latest['Close_Price']:,.2f}")

    # 3. The Data Table
    st.write("### Recent Activity Log")
    # Show newest first
    st.dataframe(df.sort_values(by='Date', ascending=False).head(10))
    
    # 4. Visuals
    st.write("### Market Regime History")
    st.bar_chart(df.set_index('Date')['Regime'])
    
except FileNotFoundError:
    st.warning("⚠️ No trade log found yet.")
    st.info("The GitHub Action needs to run at least once to generate data.")
    st.markdown("Go to **GitHub > Actions > Daily Trading Bot > Run workflow** to start it now.")