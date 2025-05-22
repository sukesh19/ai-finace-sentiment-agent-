import os
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static')

# --- Configuration ---
DATA_FOLDER = 'stock_data' # Folder where your CSV files are located

# --- Helper Functions ---

def get_nifty50_stocks_from_folder():
    """Reads stock names from CSV files in the DATA_FOLDER."""
    stock_names = []
    if os.path.exists(DATA_FOLDER):
        for filename in os.listdir(DATA_FOLDER):
            if filename.endswith(".csv"):
                stock_names.append(filename.replace(".csv", ""))
    return sorted(stock_names)

def load_historical_data(stock_name):
    """Loads historical data from a local CSV file."""
    file_path = os.path.join(DATA_FOLDER, f"{stock_name}.csv")
    if os.path.exists(file_path):
        try:
            # Read CSV, assuming 'Date' is already parsed by the update script
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            
            # Ensure 'close' column exists and is numeric (lowercase as per user's script output)
            required_cols = ['close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV for {stock_name} must contain '{' and '.join(required_cols)}' columns.")
            
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # Check for 'volume' column (lowercase) and include it if present
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df = df[['close', 'volume']]
            else:
                df = df[['close']] # Only close if volume is not there
            
            df.dropna(subset=['close'], inplace=True) # Drop rows where close price is not a number
            return df
        except Exception as e:
            print(f"Error loading historical data for {stock_name}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def fetch_realtime_data(stock_name):
    """Fetches real-time historical data from Yahoo Finance."""
    try:
        # yfinance uses ticker symbols, which often have '.NS' for NSE stocks
        ticker_symbol = f"{stock_name}.NS"
        # Fetch data for a reasonable period, e.g., last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2) # Fetch last 2 years of data
        df = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if not df.empty and 'Close' in df.columns: # yfinance returns 'Close' (uppercase)
            # Include 'Volume' if available from Yahoo Finance (uppercase)
            cols_to_include = ['Close']
            if 'Volume' in df.columns:
                cols_to_include.append('Volume')
            
            # Rename columns to lowercase to match user's local CSV format
            df_selected = df[cols_to_include].copy() # Use .copy() to avoid SettingWithCopyWarning
            df_selected.columns = [col.lower() for col in df_selected.columns]
            return df_selected
        else:
            print(f"No real-time data found for {stock_name} ({ticker_symbol}) from Yahoo Finance.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching real-time data for {stock_name}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculates common technical indicators (SMA, EMA) and handles Volume."""
    # Ensure 'close' column is present and numeric (now expecting lowercase 'close')
    if 'close' not in df.columns:
        return df # Should not happen if previous steps are correct

    # Simple Moving Averages (SMA) based on 'close'
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Exponential Moving Averages (EMA) based on 'close'
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Handle 'volume': fill NaN with 0 and scale it (now expecting lowercase 'volume')
    if 'volume' in df.columns:
        df['volume'].fillna(0, inplace=True) # Fill any NaN volumes with 0
        if df['volume'].max() > 0: # Avoid division by zero if all volumes are 0
            df['Volume_Scaled'] = df['volume'] / df['volume'].max()
        else:
            df['Volume_Scaled'] = 0
    else:
        df['Volume_Scaled'] = 0 # Add a placeholder if volume is not present at all

    # Drop rows with NaN values introduced by moving averages (first few rows)
    df.dropna(inplace=True)
    return df

def combine_and_preprocess_data(stock_name):
    """Combines local and real-time data, handles duplicates, sorts, and calculates indicators."""
    local_df = load_historical_data(stock_name)
    realtime_df = fetch_realtime_data(stock_name)

    if local_df.empty and realtime_df.empty:
        return pd.DataFrame(), "No data available for this stock from local files or Yahoo Finance."

    # Combine dataframes
    # Use concat and drop_duplicates to handle overlapping dates, keeping the most recent (Yahoo Finance)
    combined_df = pd.concat([local_df, realtime_df])

    # Remove duplicates based on index (Date), keeping the last (which would be from realtime_df if dates overlap)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

    # Sort by date
    combined_df = combined_df.sort_index()

    # Reset index to get 'Date' as a column for easier processing later
    combined_df = combined_df.reset_index()
    combined_df.rename(columns={'index': 'Date'}, inplace=True)

    # Calculate technical indicators
    combined_df = calculate_technical_indicators(combined_df)

    if combined_df.empty:
        return pd.DataFrame(), "Combined data is empty after processing or after calculating indicators. This might happen if there's not enough data for indicator calculations (e.g., less than 26 days for EMA_26)."

    return combined_df, None

def train_and_predict(data_df):
    """Trains a Linear Regression model and predicts tomorrow's price."""
    if data_df.empty or len(data_df) < 2:
        return None, "Not enough data to train the model. Need at least 2 data points after indicator calculations."

    # Convert Date to numerical format (days since first date)
    data_df['Date_Numeric'] = (data_df['Date'] - data_df['Date'].min()).dt.days

    # Define features for the model, including technical indicators and scaled volume
    features = ['Date_Numeric', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26']
    if 'Volume_Scaled' in data_df.columns:
        features.append('Volume_Scaled')

    # Ensure all features exist in the DataFrame
    for feature in features:
        if feature not in data_df.columns:
            return None, f"Missing required feature for training: {feature}. This might indicate an issue with data processing or insufficient historical data for indicator calculation."

    X = data_df[features]
    y = data_df['close']  # Target variable is 'close'

    model = LinearRegression()
    model.fit(X, y)

    # Predict for tomorrow
    last_date_numeric = X['Date_Numeric'].max()
    tomorrow_date_numeric = last_date_numeric + 1

    # Use the last known values of the indicators for prediction
    last_known_features = X.iloc[-1].copy()
    last_known_features['Date_Numeric'] = tomorrow_date_numeric
    prediction_input = np.array([last_known_features[f] for f in features]).reshape(1, -1)

    predicted_price = model.predict(prediction_input)[0]

    # Calculate accuracy if the actual price for tomorrow exists
    if len(data_df) > 1:
        actual_price = y.iloc[-1]  # Use the last known actual price
        accuracy = 100 - abs((predicted_price - actual_price) / actual_price * 100)
    else:
        accuracy = None

    return max(0, predicted_price), accuracy

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/stocks')
def api_stocks():
    """Returns a list of available stock names."""
    stocks = get_nifty50_stocks_from_folder()
    return jsonify(stocks)

@app.route('/api/data/<stock_name>')
def api_data(stock_name):
    """Fetches and returns combined historical and real-time data for a stock."""
    data_df, error_msg = combine_and_preprocess_data(stock_name)
    if error_msg:
        return jsonify({"error": error_msg}), 400

    # Convert DataFrame to list of dictionaries for JSON response
    # Ensure Date is formatted asYYYY-MM-DD
    # Only send 'Date' and 'close' for the chart, as frontend doesn't visualize indicators directly
    data_list = data_df[['Date', 'close']].to_dict(orient='records') # Use 'close'
    for item in data_list:
        item['Date'] = item['Date'].strftime('%Y-%m-%d')

    return jsonify(data_list)

@app.route('/api/predict/<stock_name>')
def api_predict(stock_name):
    """Predicts tomorrow's price for a given stock."""
    data_df, error_msg = combine_and_preprocess_data(stock_name)
    if error_msg:
        return jsonify({"error": error_msg}), 400

    predicted_price, accuracy = train_and_predict(data_df)
    if predicted_price is None:
        return jsonify({"error": accuracy}), 400

    return jsonify({
        "predictedPrice": predicted_price,
        "accuracy": f"{accuracy:.2f}%" if accuracy is not None else "N/A"
    })

@app.route('/api/analyze/<stock_name>')
def api_analyze(stock_name):
    """Fetches and returns basic analysis for a given stock."""
    data_df, error_msg = combine_and_preprocess_data(stock_name)
    if error_msg:
        return jsonify({"error": error_msg}), 400

    # Perform basic analysis
    analysis = {
        "mean_close": data_df['close'].mean(),
        "median_close": data_df['close'].median(),
        "max_close": data_df['close'].max(),
        "min_close": data_df['close'].min(),
        "total_volume": data_df['volume'].sum() if 'volume' in data_df.columns else "N/A",
    }

    return jsonify(analysis)

if __name__ == '__main__':
    # Create data folder if it doesn't exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created data folder: {DATA_FOLDER}. Please place your CSV files here.")

    # Check if any CSV files are present
    if not get_nifty50_stocks_from_folder():
        print(f"WARNING: No CSV files found in the '{DATA_FOLDER}' folder. Please add your Nifty 50 stock CSVs.")

    # Run the Flask app
    app.run(debug=True, port=5000) # You can change the port if 5000 is in use
