# 📈 Stock Market Analysis & Prediction using LSTM

![Banner](https://github.com/Roxtop07/Stock-Price-Predictor/blob/main/Blog%20Banner%20Design.jpg)

## 🧾 Overview

This project is a comprehensive implementation of stock price prediction using Long Short-Term Memory (LSTM) neural networks. Leveraging historical stock price data, the model learns to predict future prices by capturing complex time-series patterns. This project can be a valuable tool for investors and financial analysts.

---

## 🎯 Problem Statement

Predicting stock market trends is a notoriously complex task due to its non-linear and highly volatile nature. Traditional regression models often fall short. This project aims to:

* Build a deep learning model (LSTM) to forecast future stock prices.
* Compare LSTM performance with traditional machine learning models.
* Help investors make data-driven decisions.

---

## 🧠 Project Objectives

* Collect and clean historical stock data.
* Visualize patterns and price movements.
* Train and evaluate LSTM-based time-series model.
* Forecast stock prices with high accuracy.

---

## 🧰 Tools & Technologies

| Category         | Tools & Libraries   |
| ---------------- | ------------------- |
| Language         | Python              |
| Data Processing  | NumPy, Pandas       |
| Visualization    | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn        |
| Deep Learning    | TensorFlow, Keras   |
| Environment      | Jupyter Notebook    |

---

## 🧪 Workflow

### 1. Data Preprocessing

* Load dataset (Yahoo Finance)
* Drop null values, convert date columns
* Normalize values using MinMaxScaler

### 2. Feature Engineering

* Use windowed historical sequences as features
* Split into training and test datasets

### 3. LSTM Model Architecture

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
```

### 4. Training & Evaluation

* Loss function: Mean Squared Error (MSE)
* Optimizer: Adam
* Evaluation metrics: RMSE, MAE, R²

### 5. Predictions & Visualization

* Plot predicted vs actual stock prices
* Evaluate on unseen test data

---

## 🧾 Results

| Model             | R² Score | MAE  | RMSE |
| ----------------- | -------- | ---- | ---- |
| Linear Regression | 0.67     | 6.25 | 8.42 |
| Random Forest     | 0.84     | 4.02 | 5.67 |
| XGBoost Regressor | 0.89     | 3.12 | 4.29 |

> 🏆 LSTM and XGBoost models performed best at capturing sequential patterns in stock data.

![Companies Volumes](https://github.com/Roxtop07/Stock-Price-Predictor/blob/main/Stock%20Price%20Prediction%20Companies%20Volume.jpg)

---

## 📊 Visual Results

### 📌 Closing Price Trend

![Closing Price](https://github.com/Roxtop07/Stock-Price-Predictor/blob/main/Stock%20Price%20Prediction%20Project.jpg)

### 🧠 Open Close Trend

![Loss Curve](https://github.com/Roxtop07/Stock-Price-Predictor/blob/main/Stock%20Price%20Prediction%20Companies%20Open%20Close.jpg)
---

## 🔮 Future Enhancements

* Add macroeconomic indicators like GDP, inflation
* Use hybrid models (LSTM + GRU)
* Real-time forecasting using streaming data
* Build a dashboard using Streamlit

---

## 📜 Acknowledgements

* Guide: **Dr. Subhashini M.E., Ph.D.**
* Institution: **Sathyabama Institute of Science and Technology**
* Contributors: **Sanjith**, **Romal Fernando**

---

## 📘 License

This project is built for academic purposes only. Not intended for commercial or financial advice.

> Made with ❤️ for the Final Project of Elevate Labs

![Thank You](https://media.giphy.com/media/13ZHjidRzoi7n2/giphy.gif)
