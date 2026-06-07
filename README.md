# 📈 AI Stock Anomaly Detection

**Major Project | Temporal Train/Test Split | Unsupervised Machine Learning**

A modern Streamlit dashboard for detecting anomalies in stock price data using Isolation Forest, One-Class SVM, and LSTM Autoencoder with proper **temporal (time-series) validation** to avoid look-ahead bias.

![Demo](https://via.placeholder.com/800x400?text=Project+Dashboard+Screenshot)

## ✨ Features

- **Temporal Train/Test Split** (No data leakage)
- Multiple Anomaly Detection Models:
  - Isolation Forest
  - One-Class SVM
  - LSTM Autoencoder (Optional)
- Interactive **Candlestick Chart** with anomaly highlighting
- **Technical Indicators** (RSI, MACD, Bollinger Bands)
- Feature Distributions + Correlation Heatmaps
- Explainability & AI-based anomaly interpretation
- Backtesting Simulation (Strategy vs Buy & Hold)
- Multi-Stock Comparison with correlation analysis
- Model persistence with joblib + caching for fast reloads

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **ML Models**: scikit-learn, TensorFlow (LSTM)
- **Data**: yfinance
- **Others**: Pandas, NumPy, Joblib

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/sharathkalvakolla/ai-stock-anomaly.git
cd ai-stock-anomaly

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app/streamlit_app.py




## Features

- 📊 Interactive Streamlit dashboard with 8 analysis tabs
- 🧠 Three anomaly models: Isolation Forest, One-Class SVM, LSTM Autoencoder
- 🔎 AI anomaly explainer with feature-level reason codes and severity badge
- 📈 Technical indicators: RSI, MACD, Bollinger Bands, volatility
- 🧪 Backtesting simulation (educational) vs buy-and-hold baseline
- 🏷️ Multi-stock comparison with anomaly rates and return correlation heatmap

## How To Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## How To Run With Docker

```bash
docker build -t stock-anomaly-ai .
docker run --rm -p 8501:8501 stock-anomaly-ai
```

## Project Structure
```
stock-anomaly-ai/
├── app/
│   └── streamlit_app.py          # Main Dashboard
├── src/
│   ├── config.py
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── anomaly_detection.py
│   ├── visualization.py
│   └── ...
├── models/                        Saved models
├── outputs/                       Results
├── data/
├── requirements.txt
└── README.md
```

## Limitations

- Educational tool only; not financial advice or a production trading system.
- Backtesting is simplified and excludes slippage, fees, and market impact.
- Unsupervised anomalies are statistical outliers, not guaranteed trade signals.
- LSTM model availability depends on TensorFlow installation/runtime.

## Tech Stack

- Python, pandas, numpy, scikit-learn, TensorFlow/Keras
- Streamlit, Plotly, matplotlib
- yfinance, joblib, Docker
