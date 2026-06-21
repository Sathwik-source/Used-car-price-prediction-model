# Used Car Resale Price Predictor

A machine learning web application that predicts the resale price of used cars based on vehicle specifications, ownership history, and market factors. Built with Python and deployed via Streamlit.

## Live

🔗 [https://used-car-price-prediction-model-4aq2yqjgbstkhqxsrpp5kd.streamlit.app](https://used-car-price-prediction-model-4aq2yqjgbstkhqxsrpp5kd.streamlit.app)

## About the Project

This app is built using Python, Jupyter Notebook, and a combination of machine learning models — Linear Regression, Random Forests, and XGBoost. The goal is to provide an accurate, data-driven estimate of a used car's resale value based on real-world market data sourced from CarDekho.

The dataset was thoroughly preprocessed using Exploratory Data Analysis (EDA) to handle missing values, outliers, and inconsistent entries before any modelling was performed. Multiple regression models were trained and evaluated, and XGBoost was selected as the final model based on its superior R² score compared to the other approaches.

## Features

- Supports **15 Indian car brands** including Maruti, Hyundai, Tata, Honda, Toyota, Ford, and more
- Input parameters include engine displacement (cc), max power (bhp), torque (Nm), mileage, fuel type, transmission, ownership history, and purchase year
- Displays an estimated resale price along with a confidence range
- Shows feature importance so you can understand what drives the predicted price
- Responsive UI with automatic light and dark mode based on your system preference

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Modelling | Scikit-learn, XGBoost |
| Analysis | Pandas, NumPy, Jupyter Notebook |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

## How to Run Locally

```bash
git clone https://github.com/your-username/used-car-price-prediction-model.git
cd used-car-price-prediction-model
pip install -r requirements.txt
streamlit run price_app.py
```

## Project Structure

```
├── price_app.py              # Main Streamlit application
├── model_with_columns.pkl    # Trained model + column schema
├── xgb_pipeline.joblib       # XGBoost inference pipeline
└── requirements.txt          # Python dependencies
```

## Model Performance

The final XGBoost model was chosen after comparing R² scores across all three models. XGBoost achieved the best generalisation on the test set, handling the non-linear relationships between vehicle age, mileage, engine specs, and resale price more effectively than linear approaches.
