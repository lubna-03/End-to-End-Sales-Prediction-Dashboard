#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install prophet')


# In[3]:


get_ipython().system('pip install xgboost')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[5]:


train = pd.read_csv(r"C:\Users\HP\Downloads\rossmann-store-sales\train.csv")

store = pd.read_csv(r"C:\Users\HP\Downloads\rossmann-store-sales\store.csv")


# In[6]:


df = train.merge(store, on='Store', how='left')

display(df.head(3))

df.info()


# In[7]:


#basic data cleaning
df['Date']=pd.to_datetime(df['Date'])
df[df['Open']==1].copy()
df['CompetitionDistance']=df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
df['CompetitionOpenSinceYear']  = df['CompetitionOpenSinceYear'].fillna(0)
df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
df['PromoInterval'] = df['PromoInterval'].fillna('None')

print("Missing values after cleaning:")
print(df.isnull().sum().sum(), "remaining missing values")


# In[8]:


#data with pictures
plt.figure(figsize=(12,6))
sns.lineplot(x='Date',y='Sales',data=df.groupby('Date')['Sales'].mean().reset_index())
plt.title('Average Dialy Sales Over Time')
plt.show()


sns.boxplot(x='Promo',y='Sales',data=df)
plt.show()


# In[9]:


#feature enginnering
# Section 3: Feature Engineering (Key for ML Models)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Lags (e.g., sales 7 days ago)
df.sort_values(['Store', 'Date'], inplace=True)
df['Lag7'] = df.groupby('Store')['Sales'].shift(7)
df['Lag14'] = df.groupby('Store')['Sales'].shift(14)

# Fill NaNs in lags
df.fillna(0, inplace=True)  # Or better imputation


# In[10]:


#Time-series Model with p
daily_sales=df.groupby('Date')['Sales'].sum().reset_index()
daily_sales.columns=['ds','y']
m=Prophet(yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False)
m.add_country_holidays(country_name='DE')
m.fit(daily_sales)

future=m.make_future_dataframe(periods=42)
forecast=m.predict(future)


fig=m.plot(forecast)
plt.title('Prophet Forecast - Total sales')
plt.show()

fig2=m.plot_components(forecast)
plt.show()


# In[12]:


features = [
    'Store', 'DayOfWeek', 'Month', 'Year', 'Promo',
    'SchoolHoliday', 'CompetitionDistance', 'Lag7', 'Lag14'
]

X = df[features].copy()
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)
print("After cleaning X:")  # should all be int64/float64
print("Missing values:", X.isnull().sum().sum())  # should be 0

y = df['Sales']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    tree_method='hist',           # fast & more stable
    enable_categorical=False,     # ← disables categorical mode (common crash fix)
    missing=0                     # tells XGBoost 0 means missing
)


model.fit(X_train, y_train)
preds = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

xgb.plot_importance(model)
plt.title("Which features matter most?")
plt.show()


# In[16]:


print("=== MODEL COMPARISON & INSIGHTS ===")
print("XGBoost was trained on per-store daily sales (~200,000 test rows)")
print("Prophet was trained on total daily sales across all stores (~942 unique days)")
print("→ Direct MAE/RMSE comparison is not fair because the tasks are different.")
print("")
print("Strengths of each:")
print("• XGBoost: Great for predicting sales per individual store, using clues like past week sales, promotions, day of week.")
print("• Prophet: Great for seeing overall company trends, seasonal patterns, holiday effects on total sales.")
print("")
print("Business takeaway:")
print("Use BOTH:")
print("  • Prophet → plan big-picture (total demand next 6 weeks)")
print("  • XGBoost → plan per-store (which shops need more stock on Saturdays?)")
print("This combination gives the store powerful tools to reduce waste and increase sales.")


# In[ ]:





# In[ ]:




