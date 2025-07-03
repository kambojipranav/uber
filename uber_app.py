import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🚖 Uber Forecast XGBoost", layout="wide")

# -----------------------------
# CUSTOM STYLING
# -----------------------------
def set_custom_theme(theme: str = "light"):
    if theme == "dark":
        bg_color = "#111"
        text_color = "#f5f5f5"
        card_color = "#222"
        gradient = "linear-gradient(135deg, #1d2b64, #f8cdda)"
    else:
        bg_color = "#00d924"
        text_color = "#222"
        card_color = "#ffffff"
        gradient = "linear-gradient(135deg, #f6d365, #fda085)"

    st.markdown(f"""
        <style>
        html, body, .stApp {{
            background: {gradient};
            color: {text_color};
        }}
        .main {{
            background-color: {card_color};
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
        }}
        footer {{
            visibility: hidden;
        }}
        .footer-text {{
            position: fixed;
            bottom: 15px;
            left: 0;
            right: 0;
            text-align: center;
            font-size: 14px;
            color: {text_color};
            opacity: 0.8;
        }}
        </style>
        <div class="footer-text">2025 Reserved @Pranav The King</div>
    """, unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Settings")
theme_mode = st.sidebar.radio("Select Theme", ["Light", "Dark"])
set_custom_theme(theme_mode.lower())

st.sidebar.markdown("---")
lag_window = st.sidebar.slider("Lag Window Size", min_value=6, max_value=72, value=24, step=6)
run_button = st.sidebar.button("🚀 Run Forecast")

# -----------------------------
# MAIN HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🚖 Uber Trips Forecasting with XGBoost</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Stylish Time Series Forecast with Lag Features</h4>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/kambojipranav/uber/main/Uber-Jan-Feb-FOIL.csv"
    df = pd.read_csv(url)
    df.columns = [col.lower() for col in df.columns]  # make lowercase for consistency
    return df

df = load_data()

# Select default columns safely
date_col = "date"
count_col = "trips"

with st.expander("🛠 Column Selection", expanded=False):
    date_col = st.selectbox("Date/Time column", df.columns, index=df.columns.get_loc(date_col))
    count_col = st.selectbox("Trip Count column", df.columns, index=df.columns.get_loc(count_col))

# Preprocess
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col)
df.set_index(date_col, inplace=True)
df = df.resample("H").sum()
df["count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

# Show chart
st.markdown("### 📊 Trip Time Series Overview")
st.line_chart(df["count"])

# -----------------------------
# FORECASTING
# -----------------------------
if run_button:
    st.markdown("### ✅ Forecast Result")

    def create_lagged_features(series, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i + window_size])
            y.append(series[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_lagged_features(df["count"].values, lag_window)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, max_depth=6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_index = df.index[lag_window + split:]
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success(f"📊 Model Accuracy (MAPE): `{mape:.2%}`")
    st.success(f"📊 Coefficient of Determination: `{r2:.2%}`")

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_index, y=y_test, mode='lines', name='Actual', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=test_index, y=y_pred, mode='lines', name='Predicted', line=dict(color='orangered', dash='dash')))

    fig.update_layout(
        title="🚗 Uber Trips: Actual vs Predicted",
        xaxis_title="Time",
        yaxis_title="Trips",
        template="plotly_dark" if theme_mode == "Dark" else "plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Thank you for using the app! Have a great day! 🌟")
else:
    st.info("ℹ️ Click '🚀 Run Forecast' from the sidebar to generate predictions.")
