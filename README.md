# 🦎 RAAT: Regime Adaptive Algorithmic Trader

### *A Self-Correcting AI Trading System that adapts to Market Conditions.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://regime-adaptive-algorithmic-trader-msvuu8jdu842pfedsbpt4i.streamlit.app/)

## 📉 The Problem
Traditional trading algorithms fail because they assume the market always behaves the same way. A strategy that works in a **Bull Market** often loses money in a **Bear Market**.

## 🧠 The Solution: Regime Adaptation
RAAT uses a two-stage Machine Learning pipeline to adapt to changing conditions:

1.  **The "Manager" (Unsupervised Learning):**
    * **Model:** K-Means Clustering + PCA
    * **Input:** VIX, Bond Yields, Oil, Market Volatility.
    * **Output:** Detects the current "Market Regime" (e.g., *Low Volatility Bull* vs. *High Stress Bear*).

2.  **The "Specialist" (Supervised Learning):**
    * **Model:** XGBoost Classifier
    * **Logic:** We train a separate XGBoost model for *each* regime.
    * **Action:** If the Manager detects "Regime 0", it swaps in "Trader 0" to make the final Buy/Sell decision.

## 🛠️ Tech Stack
* **Core:** Python 3.9, Pandas, NumPy
* **ML:** Scikit-Learn (K-Means, PCA), XGBoost
* **Data:** Yahoo Finance API (`yfinance`)
* **Ops:** GitHub Actions (Daily Cron Job for Inference)
* **Dashboard:** Streamlit Cloud

## 🚀 How to Run Locally
1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/Niroshan-k/Regime-Adaptive-Algorithmic-Trader.git
    cd Project
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Backtest:**
    ```bash
    python main.py
    ```
4.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```

## 📊 Live Dashboard
Check out the live performance of the bot here: [\[STREAMLIT APP\]](https://regime-adaptive-algorithmic-trader-msvuu8jdu842pfedsbpt4i.streamlit.app/)

---
*Disclaimer: This project is for educational purposes only. Not financial advice.*