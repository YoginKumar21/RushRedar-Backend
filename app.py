from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Ensure paths are resolved relative to this backend folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Load environment variables from backend/.env explicitly
load_dotenv(os.path.join(ROOT_DIR, '.env'))

# Import both predictor classes
from app.utils.predictor import TrafficPredictor # This is your LSTM class
from app.utils.longpredictor import EnsemblePredictor # This is the ensemble class

app = Flask(__name__)
CORS(app) 

@app.route('/api/config', methods=['GET'])
def get_config():
   api_key = os.getenv('MY_API_KEY')
   if not api_key:
       return jsonify({"error": "API key not found"}), 500
   return jsonify({"api_key": api_key})
# --- GLOBAL MODEL LOADING ---

# Compute absolute model paths inside backend/models
MODEL_ROOT = os.path.join(ROOT_DIR, 'models')
LSTM_MODEL_DIR = os.path.join(MODEL_ROOT, 'lstm')

# Ensemble folder name may have been misspelled on disk; try both
POSSIBLE_ENSEMBLE_DIRS = ['ensemble', 'enesmble']
ENSEMBLE_MODEL_DIR = None
for d in POSSIBLE_ENSEMBLE_DIRS:
    candidate = os.path.join(MODEL_ROOT, d)
    if os.path.isdir(candidate):
        ENSEMBLE_MODEL_DIR = candidate
        break
# If none exists, default to 'models/ensemble' (will error later and be caught)
if ENSEMBLE_MODEL_DIR is None:
    ENSEMBLE_MODEL_DIR = os.path.join(MODEL_ROOT, 'ensemble')

# 1. Load Short-Term LSTM Model (from predictor.py)
try:
    print("Initializing Short-Term Classification Predictor (LSTM)...")
    short_term_predictor = TrafficPredictor(model_dir=LSTM_MODEL_DIR)
    print("✓ Short-Term Classification Predictor is ready.")
    
except Exception as e:
    print(f"FATAL: Could not load Short-Term model: {e}")
    short_term_predictor = None

# 2. Load Long-Term Trend Model (from longpredictor.py)
try:
    print("Initializing Long-Term Trend Predictor (Ensemble)...")
    long_term_predictor = EnsemblePredictor(model_path=ENSEMBLE_MODEL_DIR)
    
    # daily historical CSV now lives inside backend root
    daily_csv_path = os.path.join(ROOT_DIR, 'a.csv')
    print(f"Loading historical CSV from: {daily_csv_path}")
    daily_historical_data = pd.read_csv(daily_csv_path) 
    daily_historical_data['Date'] = pd.to_datetime(daily_historical_data['Date'], format='%d-%m-%Y')
    daily_historical_data = daily_historical_data.sort_values('Date')
    
    print("✓ Long-Term Trend Predictor is ready.")

except Exception as e:
    print(f"FATAL: Could not load Long-Term model: {e}")
    long_term_predictor = None
    daily_historical_data = None


# 3. Load the Short-Term Dataset for the API
try:
    print("Loading short-term dataset (TrafficDataset.csv)...")
    short_term_csv_path = os.path.join(ROOT_DIR, 'TrafficDataset.csv')
    print(f"Reading short-term CSV from: {short_term_csv_path}")
    short_term_df = pd.read_csv(short_term_csv_path)
    
    short_term_df['datetime'] = pd.to_datetime(short_term_df['Date'] + ' ' + short_term_df['Time'], format='%d-%m-%Y %I:%M:%S %p')
    short_term_df = short_term_df.sort_values('datetime')
    
    print(f"✓ Short-term dataset loaded. {len(short_term_df)} rows.")
    
except Exception as e:
    print(f"FATAL: Could not load TrafficDataset.csv: {e}")
    short_term_df = None

# --- API ROUTES ---

@app.route('/')
def home():
    return "Traffic Prediction API (Long & Short Term) is running."

# --- [NEW] ENDPOINT FOR RECENT DATA (FOR LSTM) ---
@app.route('/api/get-recent-data', methods=['GET'])
def get_recent_data():
    if short_term_df is None:
        return jsonify({"error": "Short-term dataset (TrafficDataset.csv) not loaded"}), 500
        
    try:
        recent_df = short_term_df.tail(48)
        data_dicts = recent_df.to_dict('records')
        return jsonify(data_dicts)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get recent data: {str(e)}"}), 500

# --- ENDPOINT 1: SHORT-TERM PREDICTION (LSTM) ---
@app.route('/api/predict', methods=['POST'])
def get_short_term_prediction():
    if not short_term_predictor:
        return jsonify({"error": "Short-term model not loaded"}), 500

    data = request.json
    if not data or 'recent_data' not in data:
        return jsonify({"error": "Missing 'recent_data' key in JSON body"}), 400
    
    recent_data_dicts = data['recent_data']
    horizon = request.args.get('horizon', '1hr') 
    
    try:
        prediction_result = short_term_predictor.predict(
            recent_data_dicts=recent_data_dicts,
            horizon_name=horizon
        )
        
        if "error" in prediction_result:
            return jsonify(prediction_result), 400
            
        return jsonify(prediction_result), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# --- ENDPOINT 2: LONG-TERM TRENDS (Ensemble) ---
@app.route('/api/trends', methods=['GET'])
def get_traffic_trends():
    if not long_term_predictor or daily_historical_data is None:
        return jsonify({"error": "Long-term model or data not loaded"}), 500

    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not start_date or not end_date:
        return jsonify({"error": "Missing 'start' or 'end' date parameters"}), 400

    base_features = {
        'Area Name': 'Indiranagar', 
        'Weather': 'Clear',   
        'Environ': 'Urban',
        'Average Speed': 55.0,
        'Congestion': 0.3,
        'Road Capacity': 3000.0,
        'Utilization': 0.6,
        'Co': 0.5 
    }

    try:
        daily_preds_df = long_term_predictor.predict_trends(
            start_date=start_date,
            end_date=end_date,
            historical_df=daily_historical_data.iloc[-60:],
            base_features=base_features
        )
        
        daily_preds_df['Date'] = pd.to_datetime(daily_preds_df['Date'])
        daily_preds_df = daily_preds_df.set_index('Date')

        weekly_trend = daily_preds_df['Predicted_Traffic'].resample('W').mean()
        monthly_trend = daily_preds_df['Predicted_Traffic'].resample('M').mean()

        response = {
            "daily": daily_preds_df.reset_index().to_dict('records'),
            "weekly": weekly_trend.reset_index().to_dict('records'),
            "monthly": monthly_trend.reset_index().to_dict('records')
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000, use_reloader=False)