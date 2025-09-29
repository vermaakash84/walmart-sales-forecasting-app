import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
import datetime

# -----------------------------
# Load saved model
# -----------------------------
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("walmart_forecast_xgb.json")  # Load the trained XGBoost model
    return model

model = load_model()

# -----------------------------
# Feature Engineering Function
# -----------------------------
def create_features(data):
    data = data.copy()
    data["weekofyear"] = data["Date"].dt.isocalendar().week.astype(int)
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year
    data["dayofweek"] = data["Date"].dt.dayofweek
    
    # Lags
    for lag in [1, 2, 3, 52]:
        data[f"lag_{lag}"] = data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)
    
    # Rolling means
    data["roll_mean_4"] = data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(window=4).mean()
    data["roll_mean_12"] = data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(window=12).mean()
    
    return data

# -----------------------------
# Recursive Forecast Function
# -----------------------------
def recursive_forecast(model, history_df, horizon=12):
    hist = history_df.copy().sort_values("Date").reset_index(drop=True)
    hist = create_features(hist).dropna().reset_index(drop=True)

    forecasts = []
    last_known_row = hist.iloc[[-1]].copy()

    feature_cols = [
        "Store_enc","Dept_enc","IsHoliday_x","Temperature","Fuel_Price",
        "CPI","Unemployment","weekofyear","month","year","dayofweek",
        "lag_1","lag_2","lag_3","lag_52","roll_mean_4","roll_mean_12"
    ]

    for step in range(horizon):
        X_last = last_known_row[feature_cols]

        dtest = xgb.DMatrix(X_last)
        y_pred = model.predict(dtest)[0]

        next_date = last_known_row["Date"].iloc[-1] + pd.Timedelta(weeks=1)

        forecasts.append({
            "Store": int(last_known_row["Store"].iloc[-1]),
            "Dept": int(last_known_row["Dept"].iloc[-1]),
            "Date": next_date,
            "Forecast": y_pred
        })

        new_row = last_known_row.iloc[-1].to_dict()
        new_row["Date"] = next_date
        new_row["Weekly_Sales"] = y_pred

        temp_hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        temp_hist = create_features(temp_hist)
        last_known_row = temp_hist.iloc[[-1]].copy()
        hist = temp_hist.copy()

    return pd.DataFrame(forecasts)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Walmart Sales Forecasting App")

st.markdown("""
This app forecasts **weekly sales** for Walmart stores and departments 
using a trained **XGBoost model**.
""")

# Sidebar inputs
store_id = st.number_input("Store ID", min_value=1, value=1)
dept_id = st.number_input("Dept ID", min_value=1, value=1)
horizon = st.slider("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12)

# File upload for history
uploaded_file = st.file_uploader("Upload historical sales CSV (Store, Dept, Date, Weekly_Sales, features...)", type=["csv"])

if uploaded_file:
    history_df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    
    # Label encode Store & Dept
    history_df["Store_enc"] = history_df["Store"]
    history_df["Dept_enc"] = history_df["Dept"]

    st.write("Uploaded History Data:", history_df.tail())

    if st.button("Generate Forecast"):
        forecast_df = recursive_forecast(model, history_df, horizon=horizon)
        st.success("✅ Forecast generated successfully!")
        st.write(forecast_df)

        # Plot
        st.line_chart(
            forecast_df.set_index("Date")["Forecast"],
            height=400,
        )

        # Download button
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file with historical sales data to start forecasting.")
