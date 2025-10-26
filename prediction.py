import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from config import *


class Predictor:

    def __init__(self):
        self.models = self._load_models()

    def _load_models(self):
        """Load latest trained models"""
        models = {}

        for model_name in ['xgb_daily', 'xgb_weekly', 'xgb_monthly']:
            model_path = MODEL_DIR / f"{model_name}_latest.pkl"
            if model_path.exists():
                models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name}")
            else:
                print(f"Warning: {model_name} not found!")

        return models

    def predict_top_n(self, df_latest):
        """Generate Top N predictions for each horizon"""

        results = {}

        for model_name, model in self.models.items():
            horizon = model_name.split('_')[1]  # daily/weekly/monthly

            # Filter by liquidity and delivery
            df_filtered = df_latest[
                (df_latest['avg_volume_20d'] >= MIN_LIQUIDITY) &
                (df_latest['delivery_pct'] >= MIN_DELIVERY_PCT)
                ].copy()

            # Remove anomalies
            df_filtered = self._apply_guardrails(df_filtered)

            if len(df_filtered) == 0:
                print(f"No stocks passed filters for {horizon}")
                continue

            # Predict probabilities
            X = df_filtered[ALL_FEATURES]
            proba = model.predict_proba(X)[:, 1]

            df_filtered['probability'] = proba
            df_filtered['model'] = model_name

            # Get Top N
            top_n = df_filtered.nlargest(TOP_N, 'probability')[
                ['SYMBOL', 'DATE', 'CLOSE', 'delivery_pct',
                 'fii_net_ma5', 'dii_net_ma5', 'probability']
            ].copy()

            top_n['horizon'] = horizon
            results[horizon] = top_n

            print(f"\n{horizon.upper()} Top {TOP_N}:")
            print(top_n.to_string(index=False))

        return results

    def _apply_guardrails(self, df):
        """Apply false-signal filters"""
        df = df.copy()

        # Volume spike without delivery
        if 'volume_ma20_ratio' in df.columns and 'delivery_pct' in df.columns:
            volume_spike = df['volume_ma20_ratio'] > 3.0
            low_delivery = df['delivery_pct'] < 40
            df = df[~(volume_spike & low_delivery)]

        return df

    def save_predictions(self, results):
        """Save predictions to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d")
        output_file = OUTPUT_DIR / f"predictions_{timestamp}.csv"

        # Combine all horizons
        if results:
            all_predictions = pd.concat(results.values(), ignore_index=True)
            all_predictions.to_csv(output_file, index=False)

            print(f"\nPredictions saved: {output_file}")

            return output_file
        else:
            print("\nNo predictions to save")
            return None