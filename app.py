import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="UAC Care Load & Placement Forecasting", layout="wide")

st.title("📊 Predictive Forecasting of Care Load & Placement Demand (UAC Program)")
st.markdown("प्रोएक्टिव रिसोर्स प्लानिंग और अर्ली-वॉर्निंग इंडिकेटर्स के लिए हेल्थकेयर एनालिटिक्स डैशबोर्ड")

st.sidebar.header("⚙️ डेटा और मॉडल सेटिंग्स")
uploaded_file = st.sidebar.file_uploader("UAC डेटासेट (CSV) अपलोड करें", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("डेमो मोड: सैंपल डेटा का उपयोग हो रहा है। अपना वास्तविक डेटा देखने के लिए CSV अपलोड करें।")
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    np.random.seed(42)
    trend = np.linspace(8000, 11500, 365)
    seasonality = 800 * np.sin(2 * np.pi * dates.dayofyear / 90)
    noise = np.random.normal(0, 200, 365)
    
    hhs_care = (trend + seasonality + noise).astype(int)
    apprehended = np.random.randint(300, 600, 365)
    cbp_custody = np.random.randint(1500, 3500, 365)
    transfers = np.random.randint(320, 580, 365)
    discharges = (transfers - np.random.randint(-40, 50, 365)).clip(min=200)
    
    df = pd.DataFrame({
        "Date": dates,
        "Children apprehended and placed in CBP custody": apprehended,
        "Children in CBP custody": cbp_custody,
        "Children transferred out of CBP custody": transfers,
        "Children in HHS Care": hhs_care,
        "Children discharged from HHS Care": discharges
    })

# 1. सुरक्षित डेटा क्लीनिंग (Safe Data Cleaning)
df.columns = [str(c).strip() for c in df.columns]
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")

df = df.groupby(date_col, as_index=True).mean(numeric_only=True)
df = df.sort_index()
df = df.asfreq("D")
df = df.interpolate(method="linear").bfill().ffill()

# कॉलम के नाम सुरक्षित रूप से पहचानना
hhs_col = [c for c in df.columns if "HHS Care" in c and "discharged" not in c.lower()][0]
dis_col = [c for c in df.columns if "discharged" in c.lower()][0]
trans_col = [c for c in df.columns if "transferred" in c.lower()][0]

# 2. Feature Engineering
df["Net_Pressure"] = df[trans_col] - df[dis_col]
df["HHS_Lag_1"] = df[hhs_col].shift(1)
df["HHS_Lag_7"] = df[hhs_col].shift(7)
df["HHS_Lag_14"] = df[hhs_col].shift(14)
df["HHS_Roll_Mean_7"] = df[hhs_col].rolling(7).mean()
df["HHS_Roll_Std_7"] = df[hhs_col].rolling(7).std()
df["DayOfWeek"] = df.index.dayofweek
df["Month"] = df.index.month

df_clean = df.dropna().copy()

target_col = st.sidebar.selectbox(
    "पूर्वानुमान टारगेट चुनें (Target Variable)",
    [hhs_col, dis_col]
)
forecast_days = st.sidebar.slider("पूर्वानुमान अवधि (Forecast Horizon - Days)", 7, 30, 14)
default_cap = int(df_clean[hhs_col].max() * 0.95) if df_clean[hhs_col].max() > 10 else 2500
capacity_threshold = st.sidebar.number_input(
    "अधिकतम शेल्टर क्षमता सीमा (Capacity Threshold)", 
    min_value=10, max_value=100000, value=default_cap
)

split_idx = int(len(df_clean) * 0.8)
train = df_clean.iloc[:split_idx]
test = df_clean.iloc[split_idx:]

features = ["HHS_Lag_1", "HHS_Lag_7", "HHS_Lag_14", "HHS_Roll_Mean_7", "HHS_Roll_Std_7", "Net_Pressure", "DayOfWeek", "Month"]
X_train, y_train = train[features], train[target_col]
X_test, y_test = test[features], test[target_col]

naive_pred = test["HHS_Lag_1"] if target_col == hhs_col else test[target_col].shift(1).bfill()

hw_model = ExponentialSmoothing(train[target_col], trend="add", seasonal="add", seasonal_periods=7).fit()
hw_pred = hw_model.forecast(len(test))

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return mae, rmse, mape

metrics_data = []
for name, preds in [("Baseline (Naive)", naive_pred), ("Statistical (Holt-Winters)", hw_pred), 
                    ("ML (Random Forest)", rf_pred), ("ML (Gradient Boosting)", gb_pred)]:
    mae, rmse, mape = get_metrics(y_test, preds)
    metrics_data.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)})

metrics_df = pd.DataFrame(metrics_data)

best_mape = metrics_df.loc[metrics_df["Model"] == "ML (Gradient Boosting)", "MAPE (%)"].values[0]
forecast_accuracy = max(0, 100 - best_mape)

full_hw = ExponentialSmoothing(df_clean[target_col], trend="add", seasonal="add", seasonal_periods=7).fit()
future_forecast = full_hw.forecast(forecast_days)
residual_std = np.std(full_hw.resid)
upper_bound = future_forecast + 1.96 * residual_std
lower_bound = future_forecast - 1.96 * residual_std

breach_days = np.where(upper_bound > capacity_threshold)[0]
breach_prob = (len(breach_days) / forecast_days) * 100
surge_lead_time = int(breach_days[0] + 1) if len(breach_days) > 0 else forecast_days
stability_index = round(100 - (residual_std / max(1, df_clean[target_col].mean()) * 100), 2)

st.subheader("🎯 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast Accuracy (%)", f"{forecast_accuracy:.2f}%")
col2.metric("Capacity Breach Risk", f"{breach_prob:.1f}%", delta="High Alert" if breach_prob > 30 else "Normal", delta_color="inverse")
col3.metric("Surge Lead Time", f"{surge_lead_time} Days" if len(breach_days) > 0 else "No Surge", "Advance Warning")
col4.metric("Forecast Stability Index", f"{stability_index}%")

if breach_prob > 0 and target_col == hhs_col:
    st.error(f"⚠️ **Early Warning Alert:** अगले **{surge_lead_time} दिनों** में HHS Care लोड निर्धारित क्षमता सीमा ({capacity_threshold:,}) को पार कर सकता है। शेल्टर और स्टाफ बढ़ाने की तैयारी करें।")
else:
    st.success("✅ **Capacity Status:** पूर्वानुमान अवधि के भीतर केयर लोड सुरक्षित सीमा के अंदर रहने की संभावना है।")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Future Forecast & Uncertainty", "🤖 Model Comparison", "🔍 EDA & Time-Series Decomposition", "⚖️ Intake vs Discharge Pressure"])

with tab1:
    st.subheader(f"अगले {forecast_days} दिनों का पूर्वानुमान ({target_col})")
    fig_fc = go.Figure()
    recent_df = df_clean.tail(60)
    fig_fc.add_trace(go.Scatter(x=recent_df.index, y=recent_df[target_col], name="Historical Data", line=dict(color="#1f77b4")))
    fig_fc.add_trace(go.Scatter(x=future_forecast.index, y=future_forecast, name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
    fig_fc.add_trace(go.Scatter(x=future_forecast.index, y=upper_bound, name="95% Upper Bound", line=dict(width=0), showlegend=False))
    fig_fc.add_trace(go.Scatter(x=future_forecast.index, y=lower_bound, name="95% Confidence Interval", fill="tonexty", fillcolor="rgba(255, 127, 14, 0.2)", line=dict(width=0)))
    if target_col == hhs_col:
        fig_fc.add_hline(y=capacity_threshold, line_dash="dot", line_color="red", annotation_text="Capacity Limit")
    st.plotly_chart(fig_fc, use_container_width=True)

with tab2:
    st.subheader("Statistical vs Machine Learning Models तुलना")
    st.dataframe(metrics_df, use_container_width=True)
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=test.index, y=y_test, name="Actual", line=dict(color="black", width=2)))
    fig_comp.add_trace(go.Scatter(x=test.index, y=hw_pred, name="Holt-Winters (Statistical)"))
    fig_comp.add_trace(go.Scatter(x=test.index, y=gb_pred, name="Gradient Boosting (ML)"))
    st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    st.subheader("Time-Series Decomposition (Trend, Seasonality, Residuals)")
    decomp = seasonal_decompose(df_clean[target_col], model="additive", period=7)
    fig_dec = px.line(x=df_clean.index, y=decomp.trend, labels={"x": "Date", "y": "Trend"}, title="Long-Term Trend Component")
    st.plotly_chart(fig_dec, use_container_width=True)

with tab4:
    st.subheader("Net Pressure Indicator (Transfers into HHS - Discharges)")
    fig_net = px.bar(df_clean.tail(90), x=df_clean.tail(90).index, y="Net_Pressure", 
                     color="Net_Pressure", color_continuous_scale="RdYlGn_r",
                     title="Daily Net Care Load Change (Positive = Crowding Increasing)")
    st.plotly_chart(fig_net, use_container_width=True)