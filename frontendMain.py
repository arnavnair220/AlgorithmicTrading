import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import joblib  # Import joblib for model loading

# Load the trained model
model = joblib.load('mainModel.joblib')

# Load the training data
train_data = pd.read_csv('train_data.csv', index_col=[0, 1])

def get_stock_recommendation():
    stock_symbol = entry_stock_symbol.get()

    try:
        # Fetch recent stock data
        stock_data = yf.download(stock_symbol, end=pd.to_datetime('today'))
        stock_data = stock_data.reset_index()

        # Feature Engineering
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
        stock_data['RSI'] = ta.rsi(stock_data['Close'], length=14)

        # MACD calculation
        macd_values = ta.macd(stock_data['Close'], fast=12, slow=26)
        stock_data['MACD'] = macd_values.iloc[:, 0]
        stock_data['MACD_Signal'] = macd_values.iloc[:, 1]

        # Bollinger Bands calculation
        bb_values = ta.bbands(stock_data['Close'])
        stock_data['BB_Lower'] = bb_values.iloc[:, 0]
        stock_data['BB_Middle'] = bb_values.iloc[:, 1]
        stock_data['BB_Upper'] = bb_values.iloc[:, 2]

        # Ensure the model is fitted with training data
        features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_Lower', 'BB_Middle', 'BB_Upper']
        X_train = train_data[features]
        y_train = train_data['Label']
        model.fit(X_train, y_train)

        # Make predictions
        X = stock_data[features].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(X)[0]

        # Display recommendation
        if prediction == 'buy':
            recommendation = "Consider buying the stock."
        elif prediction == 'sell':
            recommendation = "Consider selling the stock."
        else:
            recommendation = "Hold on to your position."

        messagebox.showinfo("Stock Recommendation", recommendation)

    except Exception as e:
        messagebox.showerror("Error", f"Error fetching stock data: {e}")

# GUI setup
root = tk.Tk()
root.title("Stock Recommendation App")

# Entry for stock symbol
label_stock_symbol = ttk.Label(root, text="Enter Stock Symbol:")
label_stock_symbol.pack(pady=10)
entry_stock_symbol = ttk.Entry(root)
entry_stock_symbol.pack(pady=10)

# Button to get recommendation
button_get_recommendation = ttk.Button(root, text="Get Recommendation", command=get_stock_recommendation)
button_get_recommendation.pack(pady=20)

# Run the GUI
root.mainloop()
