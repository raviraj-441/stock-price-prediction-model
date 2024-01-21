# Stock Price Prediction Project

## Overview
This project focuses on predicting stock prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock price data and makes predictions for future stock prices.

## Project Structure
- `Stock_Price_Prediction.ipynb`: Jupyter Notebook containing the Python code for data preprocessing, model training, and evaluation.
- `stock_price_prediction.py`: Python script version of the Jupyter Notebook code.
- `README.md`: Project documentation.

## Dependencies
- Python 3
- Libraries: NumPy, Pandas, yfinance, Matplotlib, Scikit-learn, TensorFlow, etc.

## Dataset
The project utilizes historical stock price data obtained from Yahoo Finance. The data includes Open, High, Low, Close, and Volume values.

## Model Architecture
The neural network model consists of two LSTM layers with 100 units each, followed by a Dense layer with 50 units and a final output layer. The model is compiled using the Adam optimizer and Mean Squared Error loss.

## Training
The model is trained for 150 epochs with a batch size of 32. Training and validation results are visualized using Matplotlib.

## Evaluation
The model's performance is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE). The predictions are compared against the actual stock prices.

## Prediction for the Next Day
The trained model is used to predict the stock price for the next day using the last sequence of historical data.

## Usage
1. Install the required dependencies.
2. Run the Jupyter Notebook or Python script to train the model and make predictions.

## Results(This values may differ slightly for different stocks)
- Mean Squared Error: [8.48]
- Mean Absolute Error: [2.22]

## Project Author
 Ravirajsingh Sodha

Feel free to reach out if you have any questions or feedback!
