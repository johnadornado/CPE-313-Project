import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("best_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Smart Water Consumption Forecasting")
st.write("🔮 Forecasting water usage using a trained LSTM model.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your water consumption CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp')

    st.subheader("📈 Recent Consumption (Last 7 Days)")
    st.line_chart(df.set_index("Timestamp")['Consumption_Liters'][-7*24:])

    # Forecast next 24 hours
    window = 24
    last_seq = df['Consumption_Liters'].values[-window:].reshape(-1, 1)
    scaled_last_seq = scaler.transform(last_seq)
    X_input = np.expand_dims(scaled_last_seq, axis=0)

    predictions = []
    for _ in range(24):
        pred = model.predict(X_input, verbose=0)[0]
        predictions.append(pred)
        new_input = np.append(X_input[0][1:], [pred], axis=0)
        X_input = np.expand_dims(new_input, axis=0)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    st.subheader("📊 Forecasted Consumption (Next 24 Hours)")
    future_time = pd.date_range(df['Timestamp'].iloc[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
    forecast_df = pd.DataFrame({
        'Timestamp': future_time,
        'Forecast_Liters': forecast.flatten()
    })
    st.line_chart(forecast_df.set_index('Timestamp'))

    with st.expander("📄 View Forecast Data"):
        st.dataframe(forecast_df)
else:
    st.info("Please upload a CSV file with columns 'Timestamp' and 'Consumption_Liters'.")
