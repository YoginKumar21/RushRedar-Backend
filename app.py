from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Ensure paths are resolved relative to this backend folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Load environment variables from backend/.env explicitly
load_dotenv(os.path.join(ROOT_DIR, '.env'))

# Optional: locate frontend `public` folder (sibling repo)
FRONTEND_DIR = os.path.normpath(os.path.join(ROOT_DIR, '..', 'RushRadar-FrontEnd', 'public'))

# NOTE: Delay importing heavy ML predictor modules until model-loading time
TrafficPredictor = None
EnsemblePredictor = None

app = Flask(__name__)
CORS(app) 

@app.route('/api/config', methods=['GET'])
def get_config():
    api_key = os.getenv('MY_API_KEY')
    # Return empty api_key if not provided so frontend can fallback to Leaflet
    return jsonify({"api_key": api_key or ""})
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
    # Import the heavy ML predictor here so missing ML deps don't prevent the server from starting
    try:
        from app.utils.predictor import TrafficPredictor as _TrafficPredictor
        TrafficPredictor = _TrafficPredictor
    except Exception as ie:
        print(f"Warning: could not import TrafficPredictor: {ie}")

    if TrafficPredictor is None:
        raise RuntimeError("TrafficPredictor class not available (missing dependencies)")

    short_term_predictor = TrafficPredictor(model_dir=LSTM_MODEL_DIR)
    print("✓ Short-Term Classification Predictor is ready.")
    
except Exception as e:
    print(f"FATAL: Could not load Short-Term model: {e}")
    short_term_predictor = None

# 2. Load Long-Term Trend Model (from longpredictor.py)
try:
    print("Initializing Long-Term Trend Predictor (Ensemble)...")
    # Import ensemble predictor at runtime to avoid hard dependency at module import
    try:
        from app.utils.longpredictor import EnsemblePredictor as _EnsemblePredictor
        EnsemblePredictor = _EnsemblePredictor
    except Exception as ie:
        print(f"Warning: could not import EnsemblePredictor: {ie}")

    if EnsemblePredictor is None:
        raise RuntimeError("EnsemblePredictor class not available (missing dependencies)")

    long_term_predictor = EnsemblePredictor(model_path=ENSEMBLE_MODEL_DIR)
    
    # daily historical CSV now lives inside backend root
    # Using unified CSV: `bangalore_traffic.csv`
    daily_csv_path = os.path.join(ROOT_DIR, 'bangalore_traffic.csv')
    print(f"Loading historical CSV from: {daily_csv_path}")
    daily_historical_data = pd.read_csv(daily_csv_path)
    # try expected date format first, then fallback to generic parsing
    try:
        daily_historical_data['Date'] = pd.to_datetime(daily_historical_data['Date'], format='%d-%m-%Y')
    except Exception:
        daily_historical_data['Date'] = pd.to_datetime(daily_historical_data['Date'], errors='coerce')
    # Normalize column names used by longpredictor: expect 'Traffic Volume' and 'Date'
    cols = daily_historical_data.columns.tolist()
    # prefer exact 'Traffic Volume', but accept common variants
    if 'Traffic Volume' not in cols:
        if 'Traffic_Volume' in cols:
            daily_historical_data['Traffic Volume'] = daily_historical_data['Traffic_Volume']
            print("Mapped 'Traffic_Volume' -> 'Traffic Volume' for long-term predictor.")
        elif 'Total' in cols:
            daily_historical_data['Traffic Volume'] = daily_historical_data['Total']
            print("Mapped 'Total' -> 'Traffic Volume' for long-term predictor.")
        elif 'TrafficVolume' in cols:
            daily_historical_data['Traffic Volume'] = daily_historical_data['TrafficVolume']
            print("Mapped 'TrafficVolume' -> 'Traffic Volume' for long-term predictor.")
        else:
            print("Warning: historical CSV missing 'Traffic Volume' column; long-term predictions may fail.")

    # ensure numeric and sort
    if 'Traffic Volume' in daily_historical_data.columns:
        daily_historical_data['Traffic Volume'] = pd.to_numeric(daily_historical_data['Traffic Volume'], errors='coerce').fillna(method='ffill').fillna(0)

    daily_historical_data = daily_historical_data.sort_values('Date')
    
    print("✓ Long-Term Trend Predictor is ready.")

except Exception as e:
    print(f"FATAL: Could not load Long-Term model: {e}")
    long_term_predictor = None
    daily_historical_data = None


# 3. Load the Short-Term Dataset for the API
try:
    print("Loading short-term dataset (bangalore_traffic.csv)...")
    short_term_csv_path = os.path.join(ROOT_DIR, 'bangalore_traffic.csv')
    print(f"Reading short-term CSV from: {short_term_csv_path}")
    short_term_df = pd.read_csv(short_term_csv_path)
    # Support multiple CSV schemas:
    # - If file contains 'Date' and 'Time' and 'Total' (original expected schema), use directly.
    # - If file contains 'Traffic_Volume' (new unified CSV), synthesize required columns.
    if 'Date' in short_term_df.columns and 'Time' in short_term_df.columns and 'Total' in short_term_df.columns:
        short_term_df['datetime'] = pd.to_datetime(short_term_df['Date'] + ' ' + short_term_df['Time'], format='%d-%m-%Y %I:%M:%S %p')
        short_term_df = short_term_df.sort_values('datetime')
    elif 'Traffic_Volume' in short_term_df.columns and 'Date' in short_term_df.columns:
        print("Detected unified CSV format. Synthesizing columns required for short-term predictor...")
        # create a Time column (midnight) and create coarse breakdown columns for vehicle types
        short_term_df['Time'] = '12:00:00 AM'
        # Map Traffic_Volume -> Total and split into vehicle types heuristically
        short_term_df['Total'] = short_term_df['Traffic_Volume']
        short_term_df['CarCount'] = (short_term_df['Traffic_Volume'] * 0.6).round().astype(int)
        short_term_df['BikeCount'] = (short_term_df['Traffic_Volume'] * 0.2).round().astype(int)
        short_term_df['BusCount'] = (short_term_df['Traffic_Volume'] * 0.1).round().astype(int)
        short_term_df['TruckCount'] = (short_term_df['Traffic_Volume'] * 0.1).round().astype(int)
        short_term_df['datetime'] = pd.to_datetime(short_term_df['Date'] + ' ' + short_term_df['Time'], format='%d-%m-%Y %I:%M:%S %p')
        short_term_df = short_term_df.sort_values('datetime')
    else:
        raise KeyError("CSV schema not compatible: missing expected columns ('Time'/'Total' or 'Traffic_Volume').")
    
    print(f"✓ Short-term dataset loaded. {len(short_term_df)} rows.")
    
except Exception as e:
    print(f"FATAL: Could not load bangalore_traffic.csv: {e}")
    short_term_df = None

# --- API ROUTES ---

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # If the frontend `public` folder exists next to this backend, serve it.
    if os.path.isdir(FRONTEND_DIR):
        if path == '' or path is None:
            path = 'index.html'
        try:
            return send_from_directory(FRONTEND_DIR, path)
        except Exception:
            # Fallback to index
            return send_from_directory(FRONTEND_DIR, 'index.html')

    # Otherwise fall back to a simple API root message
    return "Traffic Prediction API (Long & Short Term) is running."

# --- [NEW] ENDPOINT FOR RECENT DATA (FOR LSTM) ---
@app.route('/api/get-recent-data', methods=['GET'])
def get_recent_data():
    if short_term_df is None:
        return jsonify({"error": "Short-term dataset (bangalore_traffic.csv) not loaded"}), 500
        
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