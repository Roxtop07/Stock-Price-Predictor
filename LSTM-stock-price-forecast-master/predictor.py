import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df[['Close']]  # assuming you're predicting 'Close' price

def preprocess_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(look_back, len(scaled_data)):
        x.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

def predict(model_path, data_path):
    data = load_data(data_path)
    x, y, scaler = preprocess_data(data)

    model = load_model(model_path)
    predictions = model.predict(x)

    # Inverse transform predictions and actuals
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, color='blue', label='Actual Price')
    plt.plot(predicted_prices, color='red', label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    predict(model_path='../models/lstm_model.h5', data_path='../data/test_data.csv')