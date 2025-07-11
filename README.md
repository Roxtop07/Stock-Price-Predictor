📈 Stock Market Analysis & Prediction using LSTM

🧾 Overview

This project aims to build a machine learning-based predictive model using LSTM (Long Short-Term Memory) to forecast stock prices. It leverages historical market data and applies deep learning to identify patterns and trends to help investors make informed decisions.

⸻

🧠 Problem Statement

Stock price prediction has always been a high-stakes challenge due to the non-linear, highly dynamic nature of financial markets. This project addresses:
	•	Capturing temporal dependencies in stock price movements
	•	Applying deep learning to improve forecasting accuracy
	•	Assisting investors with intelligent, data-driven insights

⸻

🚀 Objectives
	•	Analyze and visualize historical stock data
	•	Build and train an LSTM model for time-series forecasting
	•	Evaluate model performance and visualize predictions
	•	Compare deep learning against traditional models

⸻

🗃️ Dataset
	•	Source: Yahoo Finance (e.g., IBM Stock Prices)
	•	Features Used:
	•	Open, High, Low, Close, Volume
	•	Date & Time formatted and scaled

⸻

📐 Tech Stack

Category	Tools & Libraries
Programming Language	Python
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	scikit-learn, Keras, TensorFlow
Deep Learning	LSTM (Sequential API from Keras)
IDE	Jupyter Notebook / Google Colab


⸻

🔧 Project Workflow

1. Data Preprocessing
	•	Load dataset
	•	Handle missing/null values
	•	Normalize using MinMaxScaler
	•	Create sequences for LSTM input

2. Model Building
	•	Use Sequential model from Keras
	•	Add LSTM layers + Dropout for regularization
	•	Output Dense layer to predict closing prices

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

3. Model Training & Evaluation
	•	Use mean_squared_error and RMSE for evaluation
	•	Plot Actual vs Predicted closing prices

4. Result Visualization
	•	Visualize model loss curve
	•	Plot predictions on test data


⸻

📊 Performance Metrics

Metric	Value (Example)
RMSE	4.29
MAE	3.12
R² Score	0.89

The LSTM model outperforms traditional models like Linear Regression and Random Forest in terms of capturing sequential dependencies.

⸻

💡 Future Improvements
	•	Include macroeconomic indicators (e.g., GDP, inflation, bond yield)
	•	Integrate Sentiment Analysis using financial news & tweets
	•	Use advanced architectures like GRU or Transformer-based models
	•	Deploy as a Streamlit web app for real-time prediction

⸻

📚 References
	•	Sharma, A. et al., “Survey of stock market prediction using ML”, ICECA, 2017
	•	Zhang, Z. et al., “PSO-Elman neural network for prediction”, ICSESS, 2017
	•	Kaggle & Yahoo Finance datasets

⸻

🙏 Acknowledgements

We extend sincere thanks to our guide Dr. Subhashini M.E., Ph.D., and the Department of Computer Science and Engineering, Sathyabama Institute of Science and Technology, Chennai, for their support and guidance.

⸻

📝 License

This project is developed for academic purposes and is not intended for financial or investment advice.

Made with ❤️ by Sanjith and Romal Fernando
