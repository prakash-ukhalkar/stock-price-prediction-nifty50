# Stock Price Prediction - NIFTY 50

A machine learning pipeline to forecast stock prices of NIFTY 50 companies using LSTM, Transformers, and technical indicators. This project covers everything from data acquisition, feature engineering, model building, to deployment via Streamlit.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project predicts future stock prices of NIFTY 50 companies using deep learning techniques. It follows a full ML pipeline:
- Data download & preprocessing
- Exploratory data analysis (EDA)
- Feature engineering with technical indicators
- Model training with LSTM & Transformers
- Evaluation and visualization
- Web deployment via Streamlit

---

## Project Structure

```bash
stock-price-prediction-nifty50/
│
├── README.md                 <- Project overview, setup, results
├── LICENSE                   <- MIT License
├── .gitignore                <- Ignore unnecessary files
├── requirements.txt          <- Required Python packages
│
├── data/
│   ├── raw/                  <- Raw CSVs (e.g., nifty50_2019.csv)
│   ├── processed/            <- Cleaned, merged datasets
│   └── download_data.py      <- Automated data fetching script
│
├── notebooks/
│   ├── 01_data_acquisition_and_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── ... other numbered notebooks
│
├── src/
│   ├── data_utils/           <- Data loading & cleaning logic
│   ├── features/             <- Feature engineering modules
│   ├── models/               <- Model architectures (LSTM, Transformer)
│   └── __init__.py
│
├── models/                   <- Trained model files
├── streamlit_app/            <- Deployment code
├── images/                   <- Charts & visualizations
└── docs/                     <- References, glossary

## 1. Clone the Repository
```python
git clone https://github.com/yourusername/stock-price-prediction-nifty50.git
cd stock-price-prediction-nifty50
```
## 2. Create and Activate Virtual Environment (Optional)
```python
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
## 3. Install Requirements
```python
pip install -r requirements.txt
```
## 4. Download Data
```python
python data/download_data.py
```
## Usage
### Run Jupyter Notebooks

Work through the numbered notebooks in the notebooks/ directory to explore data, engineer features, and train models.

### Train a Model (e.g., LSTM)
```python
python src/models/train_lstm.py
```
### Run Streamlit App
```python
streamlit run streamlit_app/app.py
```
