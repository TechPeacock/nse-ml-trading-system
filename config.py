from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data parameters
LOOKBACK_YEARS = 5
MIN_LIQUIDITY = 100000
MIN_DELIVERY_PCT = 30

# Label thresholds
DAILY_LABEL = {'horizon': 5, 'return_threshold': 0.03}
WEEKLY_LABEL = {'horizon': 20, 'return_threshold': 0.05}
MONTHLY_LABEL = {'horizon': 60, 'return_threshold': 0.08}

# XGBoost parameters
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# Features
PRICE_VOLUME_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_20d',
    'log_volume', 'volume_ma5_ratio', 'volume_ma20_ratio',
    'high_low_range', 'vwap_deviation',
    'obv_norm', 'nr7_flag', 'atr_norm', 'bb_width_norm'
]

SMART_MONEY_FEATURES = [
    'delivery_pct', 'delivery_zscore_20d', 'delivery_trend_5d',
    'oi_change_pct', 'oi_delta_5d',
    'bulk_deal_flag', 'block_deal_flag',
    'pcr_ratio', 'max_pain_distance'
]

FII_DII_FEATURES = [
    'fii_net_delta', 'fii_net_ma5', 'fii_net_ma20',
    'dii_net_delta', 'dii_net_ma5', 'dii_net_ma20',
    'fii_dii_divergence', 'institutional_flow_strength'
]

LIQUIDITY_FEATURES = [
    'avg_volume_20d', 'spread_proxy', 
    'market_cap_log', 'free_float_pct'
]

SECTOR_FEATURES = [
    'sector_relative_strength_5d',
    'sector_relative_strength_20d'
]

ALL_FEATURES = (PRICE_VOLUME_FEATURES + SMART_MONEY_FEATURES + 
                FII_DII_FEATURES + LIQUIDITY_FEATURES + SECTOR_FEATURES)

ANOMALY_THRESHOLDS = {
    'volume_spike_no_delivery': 3.0,
    'price_spike_no_fii': 0.05,
    'oi_surge_negative_delivery': 2.0
}

TOP_N = 10
