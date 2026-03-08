import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Page title
st.title("📈 NIFTY 50 Stock Price Prediction (LSTM)")

# Load dataset
df = pd.read_csv("NIFTY 50_Historical_PR_01011990to11102024.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Use closing prices
data = df[['Close']].values

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Load model
model = load_model("stock_model.keras", compile=False)

# Create sequences
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Split test data
split = int(len(X)*0.8)
X_test = X[split:]
y_test = y[split:]

# Predict
predictions = model.predict(X_test, verbose=0)

# Convert back to real prices
predictions_real = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y)

# Plot graph
st.subheader("Actual vs Predicted Price")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(y_test_real[-len(predictions_real):], label="Actual Price")
ax.plot(predictions_real, label="Predicted Price")
ax.legend()

st.pyplot(fig)

# Next day prediction
last_60_days = scaled_data[-60:]
last_60_days = last_60_days.reshape(1,60,1)

future_price = model.predict(last_60_days, verbose=0)
future_price_real = scaler.inverse_transform(future_price)

st.subheader("Next Day Prediction")

st.success(f"Predicted Next Day NIFTY Price: {future_price_real[0][0]:.2f}")
