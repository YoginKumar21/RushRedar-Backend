import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# This class handles the long-term ENSEMBLE model
class EnsemblePredictor:
    def __init__(self, model_path='backend/models/ensemble/'):
        """
        Loads all production artifacts for long-term trend prediction.
        Assumes models are in 'backend/models/ensemble/'
        """
        print("Loading long-term trend (ensemble) artifacts...")
        self.model_path = model_path
        
        # Load scalers
        self.scaler_X = self._load_pickle('scaler_X.pkl')
        self.scaler_y = self._load_pickle('scaler_y.pkl')
        
        # Load encoders and feature list
        self.label_encoders = self._load_pickle('label_encoders.pkl')
        self.feature_cols = self._load_pickle('feature_cols.pkl')
        
        # Load the Super Ensemble models
        self.models = {
            'xgb': self._load_pickle('XGBoost.pkl'),
            'cat': self._load_pickle('CatBoost.pkl'),
            'lgb': self._load_pickle('LightGBM.pkl'),
            'meta': self._load_pickle('super_ensemble_meta.pkl')
        }
        print("✓ Long-term artifacts loaded successfully.")

    def _load_pickle(self, filename):
        """Helper to load a pickle file."""
        try:
            with open(os.path.join(self.model_path, filename), 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            raise

    # --- Re-using your exact processing logic from the training script ---

    def _preprocess_data(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

        categorical_cols = ['Area Name', 'Weather', 'Environ']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    new_labels = set(df[col]) - set(le.classes_)
                    if new_labels:
                        le.classes_ = np.concatenate([le.classes_, list(new_labels)])
                    df[f'{col}_Encoded'] = le.transform(df[col])

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].fillna(df[col].mean())
        
        if 'Traffic Volume' in df.columns:
             df['Traffic Volume'] = df['Traffic Volume'].clip(lower=0)
                
        return df

    def _feature_engineering(self, df, window_sizes=[3, 7, 14, 30]):
        df = df.copy()
        for window in window_sizes:
            df[f'Traffic_Rolling_Mean_{window}'] = df['Traffic Volume'].rolling(window=window, min_periods=1).mean()
            df[f'Traffic_Rolling_Std_{window}'] = df['Traffic Volume'].rolling(window=window, min_periods=1).std().fillna(0)
            # ... (add all other rolling features from your script) ...

        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'Traffic_Lag_{lag}'] = df['Traffic Volume'].shift(lag)

        df['Traffic_EMA_7'] = df['Traffic Volume'].ewm(span=7, adjust=False).mean()
        df['Traffic_EMA_14'] = df['Traffic Volume'].ewm(span=14, adjust=False).mean()
        # ... (add all other features from your script: ROC, interactions, etc.) ...
        
        if 'Congestion' in df.columns and 'Average Speed' in df.columns:
            df['Congestion_Speed_Interaction'] = df['Congestion'] * df['Average Speed']
            df['Congestion_Speed_Ratio'] = (df['Congestion'] + 1) / (df['Average Speed'] + 1)
            
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
        
    def _predict_super_ensemble(self, X_scaled):
        """Helper to predict using the loaded super ensemble."""
        xgb_pred = self.models['xgb'].predict(X_scaled)
        cat_pred = self.models['cat'].predict(X_scaled)
        lgb_pred = self.models['lgb'].predict(X_scaled)
        meta_features = np.column_stack([xgb_pred, cat_pred, lgb_pred])
        scaled_prediction = self.models['meta'].predict(meta_features)
        return scaled_prediction

    def predict_trends(self, start_date, end_date, historical_df, base_features):
        date_range = pd.date_range(start_date, end_date, freq='D')
        future_df = pd.DataFrame(date_range, columns=['Date'])
        
        for col, val in base_features.items():
            future_df[col] = val

        predictions = []
        current_history = historical_df.copy()

        print(f"Generating long-term trends from {start_date} to {end_date}...")

        for date in date_range:
            row_to_predict_df = future_df[future_df['Date'] == date]
            temp_df = pd.concat([current_history, row_to_predict_df]).reset_index(drop=True)
            
            processed_df = self._preprocess_data(temp_df)
            final_features_df = self._feature_engineering(processed_df)
            
            feature_vector = final_features_df.iloc[[-1]]
            
            for col in self.feature_cols:
                if col not in feature_vector:
                    feature_vector[col] = 0
            feature_vector = feature_vector[self.feature_cols]

            scaled_features = self.scaler_X.transform(feature_vector)
            scaled_pred = self._predict_super_ensemble(scaled_features)
            final_pred = self.scaler_y.inverse_transform(scaled_pred.reshape(-1, 1))[0][0]
            
            predictions.append({'Date': date, 'Predicted_Traffic': final_pred})
            
            new_history_row = row_to_predict_df.copy()
            new_history_row['Traffic Volume'] = final_pred
            current_history = pd.concat([current_history, new_history_row])

        print("✓ Long-term trend prediction complete.")
        return pd.DataFrame(predictions)