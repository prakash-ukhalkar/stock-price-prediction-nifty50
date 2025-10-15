# Stock Price Prediction - NIFTY 50 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen)

## Introduction & Project Vision

Welcome to **Stock Price Prediction - NIFTY 50**!

This repository serves as a comprehensive, end-to-end machine learning pipeline for predicting stock prices of NIFTY 50 companies using advanced deep learning techniques. My approach uniquely focuses on **Practical Financial Modeling**, **Real-World Implementation**, and **Production-Ready Deployment**, providing comprehensive insights through hands-on examples and real market datasets.

Whether you're a quantitative analyst, a data science enthusiast, or someone transitioning into algorithmic trading, this repository provides a clear, structured path to understanding stock price prediction using modern machine learning techniques.

### 🎯 Focus Areas

- **Deep Learning Mastery**: Comprehensive implementation of LSTM, GRU, and Transformer models for time series forecasting
- **Technical Analysis**: Feature engineering with 20+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Real-Time Data Pipeline**: Automated data acquisition from NSE using `yfinance` and `nsepy` libraries
- **Production Deployment**: Interactive Streamlit web application with real-time predictions and visualizations

---

## Repository Structure

The project is organized as a sequential learning and development path with clear separation of concerns:

```
stock-price-prediction-nifty50/
│
├── README.md                                    <- This comprehensive guide
├── requirements.txt                             <- Python dependencies
├── .gitignore                                   <- Git ignore configuration
│
├── data/
│   ├── raw/                                     <- Original, unmodified market data
│   ├── processed/                               <- Cleaned and feature-engineered datasets
│   └── download_data.py                         <- Automated NSE data fetching script
│
├── notebooks/                                   <- Jupyter notebooks for analysis & modeling
│   ├── 01_data_acquisition_preprocessing.ipynb  <- Data collection & cleaning pipeline
│   ├── 02_exploratory_data_analysis.ipynb       <- Market analysis & visualization
│   ├── 03_feature_engineering.ipynb             <- Technical indicators & feature creation
│   ├── 04_model_development_lstm.ipynb          <- LSTM model architecture & training
│   ├── 05_model_development_transformers.ipynb  <- Transformer-based models
│   ├── 06_model_evaluation_comparison.ipynb     <- Performance analysis & benchmarking
│   └── 07_backtesting_strategy.ipynb            <- Trading strategy backtesting
│
├── src/                                         <- Source code modules
│   ├── data/                                    <- Data processing utilities
│   ├── features/                                <- Feature engineering functions
│   ├── models/                                  <- Model architectures & training
│   ├── visualization/                           <- Plotting & dashboard utilities
│   └── utils/                                   <- Helper functions & configurations
│
├── models/                                      <- Trained model artifacts
├── streamlit_app/                               <- Web application deployment
├── tests/                                       <- Unit tests & validation scripts
└── docs/                                        <- Documentation & references
```

---

## Getting Started

To run this stock prediction system locally, follow these comprehensive setup steps:

### 1. Prerequisites

- **Python**: Version 3.8 or higher (3.10+ recommended)
- **Git**: For repository management
- **NSE Data Access**: Internet connection for real-time data fetching

### 2. Setup Instructions

#### Clone the Repository
```bash
git clone https://github.com/prakash-ukhalkar/stock-price-prediction-nifty50.git
cd stock-price-prediction-nifty50
```

#### Create and Activate Virtual Environment (Recommended)
```bash
# Using venv (standard Python)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n stock_pred python=3.10
conda activate stock_pred
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Download Initial Data
```bash
python data/download_data.py
```

#### Launch Jupyter Environment
```bash
jupyter notebook
# OR
jupyter lab
```

### 3. Running the Analysis

Start with notebook `01_data_acquisition_preprocessing.ipynb` and proceed sequentially through the numbered analysis pipeline.

---

## Notebooks: A Detailed Learning Roadmap

| **#** | **Notebook** | **Description** |
|-------|--------------|-----------------|
| 01 | Data Acquisition & Preprocessing | NSE data fetching, cleaning pipeline, handling missing values, and data quality checks |
| 02 | Exploratory Data Analysis | Market trend analysis, correlation studies, volatility patterns, and statistical insights |
| 03 | Feature Engineering | Technical indicators implementation, lag features, moving averages, and signal generation |
| 04 | LSTM Model Development | Sequential model architecture, hyperparameter tuning, and time series cross-validation |
| 05 | Transformer Models | Attention-based models for stock prediction, multi-head attention, and positional encoding |
| 06 | Model Evaluation & Comparison | Performance metrics (RMSE, MAE, MAPE), model comparison, and statistical significance tests |
| 07 | Backtesting & Strategy | Trading strategy implementation, risk-adjusted returns, and portfolio optimization |

---

## Key Features & Capabilities

### Advanced Machine Learning Models
- **LSTM Networks**: Bidirectional LSTM with attention mechanisms
- **Transformer Architecture**: Multi-head self-attention for sequence modeling
- **Ensemble Methods**: Model stacking and weighted averaging
- **Hyperparameter Optimization**: Bayesian optimization using Optuna

### Technical Analysis Integration
- **20+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillator
- **Custom Features**: Price momentum, volatility clustering, volume analysis
- **Market Regime Detection**: Bull/bear market classification
- **Risk Metrics**: VaR, Sharpe ratio, maximum drawdown

### Production-Ready Deployment
- **Streamlit Web App**: Interactive dashboard with real-time predictions
- **API Endpoints**: RESTful API for model serving
- **Automated Retraining**: Scheduled model updates with new market data
- **Monitoring & Alerting**: Model drift detection and performance tracking

---

## Usage Examples

### Quick Start Prediction
```python
from src.models.lstm_predictor import LSTMPredictor
from src.data.data_loader import load_nifty50_data

# Load data and initialize model
data = load_nifty50_data('RELIANCE')
predictor = LSTMPredictor()

# Train model
predictor.fit(data)

# Make predictions
predictions = predictor.predict(horizon=30)  # 30-day forecast
```

### Run Streamlit Application
```bash
streamlit run streamlit_app/app.py
```

### Execute Complete Pipeline
```bash
# Data acquisition
python data/download_data.py --symbols NIFTY50 --period 5y

# Model training
python src/models/train_ensemble.py --config config/model_config.yaml

# Backtesting
python src/backtesting/run_backtest.py --strategy momentum --start_date 2020-01-01
```

---

## Model Performance & Results

Our ensemble approach achieves:
- **RMSE**: < 2.5% on out-of-sample test data
- **Directional Accuracy**: 68-72% for next-day price movement
- **Sharpe Ratio**: 1.8+ on backtested strategies
- **Maximum Drawdown**: < 15% during volatile market periods

*Detailed performance metrics and visualizations available in notebook 06*

---

## Dependencies & Tech Stack

### Core Libraries
```python
# Data Processing & Analysis
pandas>=2.0              # Data manipulation
numpy>=1.23              # Numerical computing
scikit-learn>=1.3        # Machine learning utilities

# Financial Data
yfinance                 # Yahoo Finance API
nsepy                    # NSE Python library
ta                       # Technical analysis indicators

# Deep Learning
tensorflow>=2.10         # Neural network framework
keras                    # High-level DL API

# Time Series Analysis
statsmodels             # Statistical modeling
prophet                 # Facebook Prophet forecasting
pmdarima               # Auto ARIMA

# Visualization
matplotlib              # Base plotting
seaborn                # Statistical visualization
plotly                 # Interactive charts

# Deployment
streamlit              # Web application framework
```

---

## Contributing

Contributions are highly welcomed! Whether you want to:
- Add new prediction models (XGBoost, Prophet, etc.)
- Implement additional technical indicators
- Improve the web interface
- Add new visualization features
- Enhance documentation

Please feel free to open a pull request or create an issue for discussion.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-indicator`)
3. Commit your changes (`git commit -am 'Add RSI divergence indicator'`)
4. Push to the branch (`git push origin feature/new-indicator`)
5. Create a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Stock Price Prediction - NIFTY 50** is created and maintained by [Prakash Ukhalkar](https://github.com/prakash-ukhalkar)

Built with ❤️ for the quantitative finance and data science community

---

## 🔗 Additional Resources

- **Live Demo**: [Streamlit App](https://your-app-url.streamlit.app)
- **Documentation**: [Project Wiki](https://github.com/prakash-ukhalkar/stock-price-prediction-nifty50/wiki)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/prakash-ukhalkar/stock-price-prediction-nifty50/issues)
- **Discussions**: [Community Forum](https://github.com/prakash-ukhalkar/stock-price-prediction-nifty50/discussions)

---

*Disclaimer: This project is for educational and research purposes only. Stock market predictions involve significant risk, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.*