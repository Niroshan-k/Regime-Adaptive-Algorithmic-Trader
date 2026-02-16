import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# CONFIG & STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="RAAT Dashboard", layout="wide", page_icon="🦎")


st.markdown("""
<style>
    .stMetric {
        background-color: #000000;
        padding: 15px;
        border: 1px
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR: PROJECT EXPLANATION (For Recruiters)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🦎 About RAAT")
    st.info(
        """
        **Regime Adaptive Algorithmic Trader**
        
        This system uses **Unsupervised Learning (K-Means)** to detect market regimes 
        and **Supervised Learning (XGBoost)** to predict daily moves.
        """
    )
    
    st.write("### 🧠 How it works")
    st.markdown("""
    1. **Fetch Data:** Pulls live S&P 500, VIX, & Bond Yields.
    2. **Detect Regime:**
       - 🟢 **Regime 0:** Bull Market (Low Volatility)
       - 🔴 **Regime 1:** Bear Market (High Volatility)
    3. **Predict:** XGBoost predicts direction (Up/Down).
    4. **Trade:** - If Bull: **Buy & Hold**
       - If Bear: **Short / Cash**
    """)
    
    st.write("---")
    st.caption("Built with Python, K-Means, PCA, XGBoost, GitHub Actions & Streamlit.")

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
st.title("🦎 Regime Adaptive Trading Algorithm")
st.markdown("### 🔴 Live Market Status")

# 1. LOAD DATA
try:
    df = pd.read_csv('trade_log.csv')
    
    # 2. PARSE LATEST DATA
    latest = df.iloc[-1]
    
    # --- TRANSLATE ROBOT SPEAK TO HUMAN SPEAK ---
    # NOTE: Check your own training! Usually 0=Stable, 1=Volatile.
    regime_name = "🟢 Bull / Stable" if latest['Regime'] == 0 else "🔴 Bear / Volatile"
    
    # Color logic for Signal
    signal_color = "normal"
    if "BUY" in latest['Final_Signal']:
        signal_color = "off" # Streamlit metric color trick (or use delta)
    
    # 3. METRICS ROW
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📢 AI Recommendation", 
            value=latest['Final_Signal'], 
            delta="Active Signal"
        )
        
    with col2:
        st.metric(
            label="🌍 Market Regime", 
            value=regime_name,
            delta=f"Cluster {latest['Regime']}"
        )
        
    with col3:
        st.metric(
            label="💰 S&P 500 Price", 
            value=f"${latest['Close_Price']:,.2f}",
            delta="Latest Close"
        )

    # 4. DATA TABLE
    st.divider()
    st.subheader("📜 Trade History Log")
    st.dataframe(
        df.sort_values(by='Date', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # 5. CHARTS (Only show if we have history)
    st.divider()
    st.subheader("📊 Regime & Signal History")
    
    if len(df) < 2:
        st.info("⚠️ **Waiting for more data.** The chart will appear after the bot runs for a few days.")
        # Show a placeholder image or explanation instead of a broken chart
        st.progress(10, text="Building historical database... (1/10 days collected)")
    else:
        # If we have data, show a nice chart
        # We map 0 and 1 to names for the chart
        df['Regime_Name'] = df['Regime'].map({0: 'Bull', 1: 'Bear'})
        
        # Simple Streamlit Chart
        st.bar_chart(df.set_index('Date')['Regime'])

except FileNotFoundError:
    st.error("⚠️ No data found!")
    st.warning("The GitHub Action hasn't run yet. Please run the workflow manually in GitHub Actions.")