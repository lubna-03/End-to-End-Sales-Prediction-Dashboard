# app.py - Store Sales Forecasting Dashboard
# Professional version for portfolio

import streamlit as st
import pandas as pd
import plotly.express as px

# ─── Page config ───
st.set_page_config(
    page_title="Store Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Modern styling ───
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    h1 { color: #111827; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { background-color: #2563eb; color: white; border-radius: 6px; padding: 10px 20px; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
    </style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shop.png", width=80)
    st.title("Sales Forecasting")
    st.markdown("**Time-Series & Machine Learning Powered**")
    st.markdown("Built by Lubna | Portfolio Project | Bengaluru | 2026")
    
    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown("""
    • Python, Pandas  
    • Prophet (time-series forecasting)  
    • XGBoost (gradient boosting)  
    • Streamlit (interactive UI)  
    """)

# ─── Main Title ───
st.title("📈 Store Sales Forecasting Dashboard – Time-Series & ML Powered")
st.markdown("Future sales predictions using **Prophet** on the Rossmann Store Sales dataset")

# ─── Intro text ───
st.markdown("""
This dashboard demonstrates predictive analytics for retail demand forecasting.  
It helps businesses anticipate sales, optimize inventory, reduce waste, and plan staffing & promotions more effectively.
""")

# ─── Upload section ───
st.subheader("Upload Your Sales Data (Optional)")
uploaded_file = st.file_uploader(
    "Upload CSV with 'Date' and 'Sales' columns",
    type="csv",
    help="Skip to see sample forecast"
)

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Data Preview (first 10 rows):")
        st.dataframe(user_data.head(10))
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("No file uploaded → showing sample forecast below.")

# ─── Load forecast ───
@st.cache_data
def load_forecast():
    try:
        df = pd.read_csv("prophet_forecast.csv")
        df['ds'] = pd.to_datetime(df['ds'])
        return df, True
    except FileNotFoundError:
        dates = pd.date_range(start='2015-08-01', periods=42, freq='D')
        dummy = pd.DataFrame({
            'ds': dates,
            'yhat': [45000 + i*350 for i in range(42)],
            'yhat_lower': [42000 + i*350 for i in range(42)],
            'yhat_upper': [48000 + i*350 for i in range(42)]
        })
        return dummy, False

forecast, real_data_loaded = load_forecast()

# ─── Forecast display ───
st.subheader("Future Sales Predictions (Next 42 Days)")

if real_data_loaded:
    st.success("Loaded your saved Prophet forecast!")
else:
    st.info("prophet_forecast.csv not found → showing example data")

st.dataframe(
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14).style.format({
        'ds': '{:%Y-%m-%d}',
        'yhat': '{:,.0f}',
        'yhat_lower': '{:,.0f}',
        'yhat_upper': '{:,.0f}'
    }),
    use_container_width=True
)

# ─── Interactive chart ───
st.subheader("Forecast Visualization")
fig = px.line(
    forecast,
    x='ds',
    y='yhat',
    title="Predicted Daily Sales with Uncertainty Range",
    labels={'ds': 'Date', 'yhat': 'Predicted Sales'},
    color_discrete_sequence=["#4f46e5"]
)

fig.add_scatter(
    x=forecast['ds'], y=forecast['yhat_upper'],
    mode='lines', line=dict(width=0), showlegend=False
)
fig.add_scatter(
    x=forecast['ds'], y=forecast['yhat_lower'],
    mode='lines', fill='tonexty',
    fillcolor='rgba(79, 70, 229, 0.18)',
    line=dict(width=0),
    name='Uncertainty'
)

fig.update_layout(
    height=550,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Predicted Total Sales (₹)"
)

st.plotly_chart(fig, use_container_width=True)

# ─── Download ───
st.subheader("Download Forecast Data")
csv = forecast.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Predictions CSV",
    data=csv,
    file_name="rossmann_sales_forecast.csv",
    mime="text/csv"
)

# ─── Footer ───
st.markdown("---")
st.caption("Built with Streamlit, Prophet & Pandas  | Lubna Shimreen")