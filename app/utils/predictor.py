import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

class TrafficPredictor:
    def __init__(self, model_dir='backend/models/lstm'):
        print("Loading traffic prediction models...")
        self.model_dir = model_dir
        
        # 1. Load Metadata
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.seq_length = self.metadata['seq_length']
        self.features = self.metadata['features']
        self.horizons = self.metadata['horizons']
        
        # 2. Load Scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # 3. Load Models
        self.models = {}
        for horizon_name in self.horizons.keys():
            model_path = os.path.join(self.model_dir, f'traffic_model_{horizon_name}.h5')
            # Use compile=False for faster loading during inference
            self.models[horizon_name] = load_model(model_path, compile=False)
            
        print("✓ Models loaded successfully.")

    def _preprocess_inference_data(self, df):
        """Applies the same feature engineering as the training script."""
        
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
             df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %I:%M:%S %p')
             df.set_index('datetime', inplace=True)
        
        # 1. Feature Engineering
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                             (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)

        # 2. Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # 3. Rolling statistics
        df['total_rolling_mean_6'] = df['Total'].rolling(window=6, min_periods=1).mean()
        df['total_rolling_std_6'] = df['Total'].rolling(window=6, min_periods=1).std().fillna(0)
        
        # 4. Fill NaNs that might appear from rolling means (at the beginning)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True) # Fill any remaining NaNs

        # 5. Select and reorder features
        try:
            return df[self.features]
        except KeyError as e:
            print(f"Error: Missing feature {e}. Available columns: {df.columns.tolist()}")
            raise
    
    def _inverse_transform_prediction(self, scaled_pred):
        """Inverse transforms the scaled prediction back to the original value."""
        dummy_array = np.zeros((1, len(self.features)))
        try:
            total_col_index = self.features.index('Total')
        except ValueError:
            total_col_index = 4 # Fallback to index 4 if not found
        
        dummy_array[0, total_col_index] = scaled_pred
        unscaled_array = self.scaler.inverse_transform(dummy_array)
        return unscaled_array[0, total_col_index]

    def _get_classification(self, count):
        """Classifies traffic count into categories. Adjust thresholds as needed."""
        count = float(count)
        if count < 100:
            return "Light"
        elif count < 200:
            return "Normal"
        elif count < 300:
            return "Heavy"
        else:
            return "Extreme"

    def predict(self, recent_data_dicts, horizon_name='1hr'):
        """
        Makes a traffic prediction.
        
        :param recent_data_dicts: A list of dicts, e.g., from request.json()
                                  Must contain at least `seq_length` data points.
        :param horizon_name: One of '15min', '1hr', '2hr', '3hr'
        :return: A dictionary with the prediction
        """
        
        if horizon_name not in self.models:
            return {"error": f"Invalid horizon. Choose from {list(self.models.keys())}"}

        # 1. Convert input data to DataFrame
        df = pd.DataFrame(recent_data_dicts)
        
        # We need at least `seq_length`
        if len(df) < self.seq_length:
            return {"error": f"Not enough data. Need at least {self.seq_length} data points, got {len(df)}."}

        # 2. Preprocess the data
        try:
            processed_df = self._preprocess_inference_data(df)
        except Exception as e:
            return {"error": f"Data preprocessing failed: {str(e)}"}
        
        # 3. Get the last `seq_length` steps for the model
        last_sequence = processed_df.tail(self.seq_length)
        
        if len(last_sequence) < self.seq_length:
             return {"error": "Processed data is still less than sequence length. Check input."}
        
        # 4. Scale the sequence
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # 5. Reshape for LSTM: (1, seq_length, n_features)
        model_input = np.array([scaled_sequence])
        
        # 6. Make prediction
        model = self.models[horizon_name]
        scaled_pred = model.predict(model_input)[0][0]
        
        # 7. Inverse transform the prediction
        final_prediction = self._inverse_transform_prediction(scaled_pred)
        
        # 8. Get classification
        classification = self._get_classification(final_prediction)
        
        # 9. Return the complete result
        return {
            "prediction_horizon": horizon_name,
            "predicted_total_traffic": round(float(final_prediction), 2),
            "traffic_classification": classification # This is the key
        }