import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

from config import *
from data_loader import NSEDataLoader
from feature_engineering import FeatureEngine
from model_training import ModelTrainer
from prediction import Predictor
from data_quality import DataQualityChecker
from partial_data_handler import PartialDataHandler

def check_data_availability():
    """Check if manual downloads are present"""
    loader = NSEDataLoader()
    
    print("\n" + "="*60)
    print("DATA AVAILABILITY CHECK")
    print("="*60)
    
    checks = {
        'Bhavcopy': loader.bhav_dir,
        'Delivery': loader.delivery_dir,
        'FII/DII': loader.fii_dii_dir,
        'Participant OI': loader.participant_dir,
        'Bulk/Block': loader.bulk_block_dir
    }
    
    all_good = True
    
    for name, path in checks.items():
        files = list(path.glob("*"))
        if files:
            print(f"✓ {name:20s}: {len(files)} files found")
        else:
            print(f"✗ {name:20s}: NO FILES FOUND")
            print(f"  → Please download and place in: {path}")
            all_good = False
    
    print("="*60 + "\n")
    
    if not all_good:
        print("⚠️  Missing data files! Please download manually from NSE.")
        return False
    
    return True

def post_market_routine():
    """Post-market: Load manual downloads and retrain models"""
    print("\n" + "="*60)
    print("POST-MARKET ROUTINE: PROCESSING MANUAL DOWNLOADS")
    print("="*60 + "\n")
    
    if not check_data_availability():
        sys.exit(1)
    
    print("Loading data from manual downloads...")
    loader = NSEDataLoader()
    df = loader.load_all_data()
    
    if df.empty:
        print("ERROR: No data loaded! Please check manual downloads.")
        sys.exit(1)
    
    print("\nRunning data quality checks...")
    qc = DataQualityChecker(df)
    issues, warnings, stats = qc.run_all_checks()
    
    if issues:
        print("\n⚠️  Critical data quality issues detected!")
        response = input("\nContinue despite issues? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please fix data issues and re-run.")
            sys.exit(1)
    
    pdh = PartialDataHandler(df)
    missing_dates = pdh.detect_missing_dates()
    
    if missing_dates:
        print(f"\n⚠️  Found {len(missing_dates)} missing trading days")
        pdh.print_coverage_summary()
        
        print("\nHow to handle missing data?")
        print("  1. Forward fill (use last known values)")
        print("  2. Interpolate (estimate missing values)")
        print("  3. Skip incomplete symbols")
        print("  4. Continue as-is (mark only)")
        
        choice = input("\nChoice (1-4) [default: 1]: ").strip() or '1'
        
        strategies = {
            '1': 'forward_fill',
            '2': 'interpolate',
            '3': 'skip',
            '4': 'mark_only'
        }
        
        strategy = strategies.get(choice, 'forward_fill')
        df = pdh.handle_missing_data(strategy=strategy)
    
    print("\nFeature Engineering...")
    fe = FeatureEngine(df)
    df = fe.create_all_features()
    
    processed_file = PROCESSED_DATA_DIR / f"processed_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(processed_file, index=False)
    print(f"\n✓ Processed data saved: {processed_file}")
    
    print(f"\nProcessing Statistics:")
    print(f"  • Total rows: {len(df):,}")
    print(f"  • Unique symbols: {df['SYMBOL'].nunique()}")
    print(f"  • Date range: {df['DATE'].min().strftime('%Y-%m-%d')} to {df['DATE'].max().strftime('%Y-%m-%d')}")
    
    print("\n" + "-"*60)
    print("Training XGBoost Models...")
    print("-"*60)
    
    print("\n### DAILY MODEL ###")
    trainer_daily = ModelTrainer(df, ALL_FEATURES, 'label_daily', 'xgb_daily')
    trainer_daily.train()
    trainer_daily.save_model()
    
    print("\n### WEEKLY MODEL ###")
    trainer_weekly = ModelTrainer(df, ALL_FEATURES, 'label_weekly', 'xgb_weekly')
    trainer_weekly.train()
    trainer_weekly.save_model()
    
    print("\n### MONTHLY MODEL ###")
    trainer_monthly = ModelTrainer(df, ALL_FEATURES, 'label_monthly', 'xgb_monthly')
    trainer_monthly.train()
    trainer_monthly.save_model()
    
    print("\n" + "="*60)
    print("✅ POST-MARKET ROUTINE COMPLETE")
    print("="*60 + "\n")
    print("Next step: Run 'python main.py predict' before market opens")

def pre_market_routine():
    """Pre-market: Generate predictions"""
    print("\n" + "="*60)
    print("PRE-MARKET ROUTINE: GENERATING PREDICTIONS")
    print("="*60 + "\n")
    
    processed_files = sorted(PROCESSED_DATA_DIR.glob("processed_*.parquet"))
    
    if not processed_files:
        print("ERROR: No processed data found. Run post-market routine first.")
        sys.exit(1)
    
    df = pd.read_parquet(processed_files[-1])
    print(f"✓ Loaded processed data: {processed_files[-1].name}")
    
    latest_date = df['DATE'].max()
    df_latest = df[df['DATE'] == latest_date].copy()
    
    print(f"  • Latest data date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  • Stocks available: {df_latest['SYMBOL'].nunique()}")
    
    print("\nGenerating predictions...")
    predictor = Predictor()
    results = predictor.predict_top_n(df_latest)
    
    output_file = predictor.save_predictions(results)
    
    print("\n" + "="*60)
    print("✅ PRE-MARKET ROUTINE COMPLETE")
    print("="*60)
    print(f"\n📊 Predictions saved: {output_file}")
    print("\nReady for trading!")

def quality_check_only():
    """Run quality checks on existing processed data"""
    processed_files = sorted(PROCESSED_DATA_DIR.glob("processed_*.parquet"))
    
    if not processed_files:
        print("ERROR: No processed data found.")
        sys.exit(1)
    
    df = pd.read_parquet(processed_files[-1])
    
    print(f"\nRunning quality checks on: {processed_files[-1].name}")
    
    qc = DataQualityChecker(df)
    qc.run_all_checks()
    
    pdh = PartialDataHandler(df)
    pdh.print_coverage_summary()

def show_predictions():
    """Display latest predictions"""
    prediction_files = sorted(OUTPUT_DIR.glob("predictions_*.csv"))
    
    if not prediction_files:
        print("No predictions found. Run 'python main.py predict' first.")
        sys.exit(1)
    
    latest_file = prediction_files[-1]
    df = pd.read_csv(latest_file)
    
    print("\n" + "="*60)
    print(f"LATEST PREDICTIONS - {latest_file.name}")
    print("="*60 + "\n")
    
    for horizon in ['daily', 'weekly', 'monthly']:
        horizon_data = df[df['horizon'] == horizon]
        if not horizon_data.empty:
            print(f"\n{horizon.upper()} Top {len(horizon_data)}:")
            print("-" * 60)
            display_cols = ['SYMBOL', 'CLOSE', 'delivery_pct', 'fii_net_ma5', 'dii_net_ma5', 'probability']
            print(horizon_data[display_cols].to_string(index=False))
            print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nNSE ML Trading System")
        print("="*60)
        print("\nUsage:")
        print("  python main.py check      # Check data availability")
        print("  python main.py train      # Process data & train models")
        print("  python main.py predict    # Generate predictions")
        print("  python main.py quality    # Run quality checks only")
        print("  python main.py show       # Show latest predictions")
        print("\nDaily Workflow:")
        print("  Evening:  Download NSE data → check → train")
        print("  Morning:  predict → show")
        print()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "check":
        check_data_availability()
    elif mode == "train":
        post_market_routine()
    elif mode == "predict":
        pre_market_routine()
    elif mode == "quality":
        quality_check_only()
    elif mode == "show":
        show_predictions()
    else:
        print(f"Unknown mode: {mode}")
        print("Run 'python main.py' for help")
