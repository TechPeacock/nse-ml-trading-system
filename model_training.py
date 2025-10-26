import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
from datetime import datetime

class ModelTrainer:
    
    def __init__(self, df, features, label_col, model_name):
        self.df = df
        self.features = features
        self.label_col = label_col
        self.model_name = model_name
        self.model = None
        
    def prepare_data(self):
        """Prepare training data with filters"""
        from config import MIN_LIQUIDITY, MIN_DELIVERY_PCT
        
        df = self.df.copy()
        df = df.dropna(subset=[self.label_col])
        df = df.dropna(subset=self.features)
        
        if 'avg_volume_20d' in df.columns:
            df = df[df['avg_volume_20d'] >= MIN_LIQUIDITY]
        
        if 'delivery_pct' in df.columns:
            df = df[df['delivery_pct'] >= MIN_DELIVERY_PCT]
        
        df = self._remove_anomalies(df)
        
        print(f"Training samples after filters: {len(df)}")
        if len(df) > 0:
            print(f"Positive class ratio: {df[self.label_col].mean():.3f}")
        
        return df
    
    def _remove_anomalies(self, df):
        """Remove potential fake signals"""
        from config import ANOMALY_THRESHOLDS
        
        df = df.copy()
        
        if 'volume_ma20_ratio' in df.columns and 'delivery_pct' in df.columns:
            volume_spike = df['volume_ma20_ratio'] > ANOMALY_THRESHOLDS['volume_spike_no_delivery']
            low_delivery = df['delivery_pct'] < 40
            df = df[~(volume_spike & low_delivery)]
        
        if 'returns_1d' in df.columns and 'fii_net_delta' in df.columns:
            price_spike = df['returns_1d'] > ANOMALY_THRESHOLDS['price_spike_no_fii']
            negative_fii = df['fii_net_delta'] < 0
            df = df[~(price_spike & negative_fii)]
        
        return df
    
    def train(self):
        """Train XGBoost model with time-series CV"""
        from config import XGB_PARAMS
        
        df = self.prepare_data()
        
        if len(df) == 0:
            print(f"ERROR: No training data available for {self.model_name}")
            return None
        
        X = df[self.features]
        y = df[self.label_col]
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        self.model = XGBClassifier(**XGB_PARAMS)
        self.model.fit(X, y)
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model_cv = XGBClassifier(**XGB_PARAMS)
            model_cv.fit(X_train, y_train)
            
            y_pred_proba = model_cv.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            cv_scores.append(score)
        
        print(f"\n{self.model_name} Training Complete")
        print(f"CV ROC-AUC Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"Mean CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        self._log_feature_importance()
        
        return self.model
    
    def _log_feature_importance(self):
        """Log top 20 important features"""
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 20 Features for {self.model_name}:")
        print(importance_df.head(20).to_string(index=False))
        
    def save_model(self):
        """Save model with timestamp"""
        from config import MODEL_DIR
        
        if self.model is None:
            print(f"ERROR: No model to save for {self.model_name}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODEL_DIR / f"{self.model_name}_{timestamp}.pkl"
        
        joblib.dump(self.model, model_path)
        
        latest_path = MODEL_DIR / f"{self.model_name}_latest.pkl"
        joblib.dump(self.model, latest_path)
        
        print(f"Model saved: {model_path}")
        
        return model_path