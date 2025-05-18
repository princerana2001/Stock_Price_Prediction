import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf


st.title("Stock Price Predictor")

stock = st.text_input("Enter the Stock ID","GOOG")
from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = google_data[['Close']].iloc[splitting_len:]


def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Input for future date prediction with a unique key
future_date = st.date_input("Enter the Future Date for Prediction", datetime.now(), key="future_date")

    # Button to trigger prediction
if st.button("Predict"):
        # Check if future_date is after the latest date in the dataset
    if future_date <= google_data.index[-1].date():
            st.error("Please choose a date in the future.")
    else:
        # Prepare the data to predict for the future date
        future_data = google_data[['Close']].tail(100)  # Use the last 100 days for prediction
        scaled_future_data = scaler.transform(future_data[['Close']])
        # Prepare the input for the model (last 100 days)
        future_x_data = []
        future_x_data.append(scaled_future_data)  # We pass the data as a 3D array
        future_x_data = np.array(future_x_data)

        # Predict the stock price for the future date
        future_prediction = model.predict(future_x_data)
        future_prediction_inv = scaler.inverse_transform(future_prediction)

        # Display predicted future price
        st.subheader(f"Predicted Stock Price for {future_date}")
        st.write(f"Predicted price: {future_prediction_inv[0][0]:.2f}")
