import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import yfinance



def fetch_or_load_smp_data():
    # Checking if the data file exists
    if os.path.exists('smp_data.csv'):
        # Load the data from the file if it exists
        smp_data = pd.read_csv('smp_data.csv', index_col=[0, 1])
    else:
        smp500Symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        smp500Symbols = (smp500Symbols.str.replace('.', '_')).unique().tolist()

        smp_data = yfinance.download(tickers=smp500Symbols, start='2014-01-01', end='2024-01-01').stack()
        smp_data.index.names = ['date', 'ticker']
        
        # Save the data to a file for future use
        smp_data.to_csv('smp_data.csv')
    
    return smp_data

smp_data = fetch_or_load_smp_data()

# Filtering data for sample stock
sample_stock = smp_data.loc[smp_data.index.get_level_values('ticker') == 'AAPL'].copy()

# Handle missing values
sample_stock.loc[:, 'Close'].fillna(method='ffill', inplace=True)

# Feature Engineering for sample stock (AAPL)
sample_stock['SMA_50'] = sample_stock['Close'].rolling(window=50).mean()
sample_stock['SMA_200'] = sample_stock['Close'].rolling(window=200).mean()
sample_stock['RSI'] = ta.rsi(sample_stock['Close'], length=14)

# Calculating MACD using pandas_ta library
macd_values = ta.macd(sample_stock['Close'], fast=12, slow=26)
sample_stock['MACD'] = macd_values.iloc[:, 0]  # MACD value
sample_stock['MACD_Signal'] = macd_values.iloc[:, 1]  # Signal line

# Calculating Bollinger Bands using pandas_ta library
bb_values = ta.bbands(sample_stock['Close'])
sample_stock['BB_Lower'] = bb_values.iloc[:, 0]  # Lower band
sample_stock['BB_Middle'] = bb_values.iloc[:, 1]  # Middle band
sample_stock['BB_Upper'] = bb_values.iloc[:, 2]  # Upper band

# Label Generation (Example: Predicting next day's price movement)
sample_stock['Next_Day_Return'] = sample_stock['Close'].shift(+1) - sample_stock['Close']
sample_stock['Next_Next_Day_Return'] = sample_stock['Close'].shift(+2) - sample_stock['Close']

def label(row):
    #if row['Next_Next_Day_Return'] > row['Next_Day_Return'] and row['Next_Next_Day_Return'] > 0:
    #    return 'hold'
    #el
    if row['Next_Day_Return'] > 0:
        return 'sell'  
    else:
        return 'buy'

sample_stock['Label'] = sample_stock.apply(label, axis=1)
sample_stock.dropna(inplace=True)

#print(sample_stock)

# Splitting Data for Training and Testing
train_size = int(0.8 * len(sample_stock))
train_data = sample_stock[:train_size]
test_data = sample_stock[-261:]

# Splitting Data for Features and Labels
features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_Lower', 'BB_Middle', 'BB_Upper']
X_train = train_data[features]
y_train = train_data['Label']
X_test = test_data[features]
y_test = test_data['Label']

# Model Building - RandomForestClassifier with Hyperparameter Tuning
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_model = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf_model, param_grid, cv=5)
rf_cv.fit(X_train, y_train)

print("Best Parameters:", rf_cv.best_params_)

# Predictions on test data
predictions = rf_cv.best_estimator_.predict(X_test)

def calculate_roi(strategy_data):
    # Step 2: Calculate Daily Returns
    strategy_data['Daily_Return'] = strategy_data['Close'].pct_change()

    # Step 4: Calculate Strategy Returns
    strategy_data['Strategy_Return'] = strategy_data['Daily_Return'] * strategy_data['Predicted_Label']

    # Step 5: Cumulative Returns
    strategy_data['Cumulative_Return'] = (1 + strategy_data['Strategy_Return']).cumprod() - 1

    # Step 6: Calculate ROI
    roi = strategy_data['Cumulative_Return'].iloc[-1] * 100
    return roi

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Strategy Implementation and Backtesting
label_map = {'buy': -1, 'sell': 1, 'hold': 0}
test_data = test_data.copy()  # Create a copy to avoid SettingWithCopyWarning
test_data['Predicted_Label'] = predictions
test_data.loc[:, 'Predicted_Label'] = test_data['Predicted_Label'].map(label_map)
test_data.loc[:, 'Strategy_Return'] = test_data['Next_Day_Return'] * test_data['Predicted_Label']
cumulative_returns = test_data['Strategy_Return'].cumsum()
cumulative_returns.index = pd.to_datetime([x[0] for x in cumulative_returns.index])

# Calculate ROI
roi = calculate_roi(test_data)
print(f"ROI: {roi:.2f}%")

# Plotting Cumulative Returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns)
plt.title('Cumulative Returns of Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

import joblib  # Import joblib for model loading
# Save the trained model
joblib.dump(rf_cv.best_estimator_, 'mainModel.joblib')

# Save the training data
train_data.to_csv('train_data.csv')
