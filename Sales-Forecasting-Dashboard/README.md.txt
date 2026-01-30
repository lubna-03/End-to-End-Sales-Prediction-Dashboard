Sales Forecasting Dashboard

End-to-end time-series sales prediction project using the Rossmann dataset.

## What I Did
- Cleaned data & created time-based features (lags, seasonality)
- Built Prophet model for aggregate trends & holidays
- Built XGBoost model for per-store predictions (MAE ≈ 891)
- Compared models and explained business impact
- Created interactive dashboard with Streamlit

## Technologies
- Python, Pandas, Scikit-learn
- Prophet, XGBoost
- Streamlit (dashboard)

## How to Run
1. Install packages:
pip install -r requirements.txt

2. Open notebook:
jupyter notebook sales_forecasting.ipynb

3. Run dashboard:

## Results
- XGBoost MAE: ~891 (per-store)
- Key drivers: Lag7 (past week sales), Promo, DayOfWeek
- Helps stores optimize stock, staff & promotions

streamlit run app.py
text## Results
- XGBoost MAE: ~891 (per-store)
- Key drivers: Lag7 (past week sales), Promo, DayOfWeek
- Helps stores optimize stock, staff & promotions