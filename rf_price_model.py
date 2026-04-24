"""
Market Price Prediction Model - Random Forest Regressor
"""
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPriceForecastModel:
    """
    Wrapper for the trained Random Forest model for market price prediction.
    Features expected by model: ['State', 'District', 'Market', 'Commodity', 'Min_Price', 'Month', 'Day']
    """
    
    def __init__(self, model_path="models/market_price_model.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()
        
        # Default categorical mappings for the demo
        # These represent typical label-encoded values for the target region
        self.mappings = {
            'state': {'Tamil Nadu': 1, 'Default': 1},
            'district': {'Tiruppur': 1, 'Default': 1},
            'market': {'Avinashi': 1, 'Default': 1},
            'commodity': {'Tomato': 1, 'Default': 1}
        }

    def _load_model(self):
        """Load the model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Market price model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Running in mock mode.")
        except Exception as e:
            logger.error(f"Error loading market price model: {e}")
            self.model = None

    def _encode_features(self, state, district, market, commodity):
        """Encode categorical features to integers for the model"""
        s_val = self.mappings['state'].get(state, self.mappings['state']['Default'])
        d_val = self.mappings['district'].get(district, self.mappings['district']['Default'])
        m_val = self.mappings['market'].get(market, self.mappings['market']['Default'])
        c_val = self.mappings['commodity'].get(commodity, self.mappings['commodity']['Default'])
        return s_val, d_val, m_val, c_val

    def predict(self, state, district, market, commodity, current_price, target_date=None):
        """
        Predict price for a specific date/features
        """
        if not self.model:
            # Fallback simple logic if model is missing
            return current_price * 1.05
            
        if target_date is None:
            target_date = datetime.now()
            
        s, d, m, c = self._encode_features(state, district, market, commodity)
        
        features = [[
            s,
            d,
            m,
            c,
            float(current_price),
            target_date.month,
            target_date.day
        ]]
        
        try:
            # Use DataFrame if model was trained with feature names to avoid warnings
            if hasattr(self.model, 'feature_names_in_'):
                df = pd.DataFrame(features, columns=self.model.feature_names_in_)
                prediction = self.model.predict(df)
            else:
                prediction = self.model.predict(features)
            
            return float(prediction[0])
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return current_price * 1.02 # Safe fallback

    def predict_price_forecast(self, state, district, market, commodity, current_price):
        """
        Generate 24h, 48h, and 72h forecasts
        """
        now = datetime.now()
        
        p_24h = self.predict(state, district, market, commodity, current_price, now + timedelta(days=1))
        p_48h = self.predict(state, district, market, commodity, current_price, now + timedelta(days=2))
        p_72h = self.predict(state, district, market, commodity, current_price, now + timedelta(days=3))
        
        return {
            "predicted_price_24h": round(p_24h, 2),
            "predicted_price_48h": round(p_48h, 2),
            "predicted_price_72h": round(p_72h, 2)
        }

if __name__ == "__main__":
    # Test script
    tester = RFPriceForecastModel()
    results = tester.predict_price_forecast("Tamil Nadu", "Tiruppur", "Avinashi", "Tomato", 20.0)
    print(f"Forecast results for 20.0/kg: {results}")
