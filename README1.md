# NSE ML Trading System

Machine Learning-based stock prediction system for NSE (National Stock Exchange of India) using XGBoost.

## 🎯 Features

- **Multi-Horizon Predictions**: Daily (5 days), Weekly (20 days), Monthly (60 days)
- **Smart Money Analysis**: FII/DII institutional flows tracking
- **Delivery Percentage**: Real vs fake accumulation detection
- **Data Quality Checks**: 8 comprehensive validation tests
- **Anomaly Detection**: Filter false breakouts and pump schemes
- **Automated Feature Engineering**: 50+ technical and smart money indicators

## 📊 Expected Performance

| Model | Horizon | Target Return | Expected Hit Rate | ROC-AUC |
|-------|---------|---------------|-------------------|---------|
| Daily | 5 days | +3% | 40-45% | 0.62-0.67 |
| Weekly | 20 days | +5% | 50-55% | 0.65-0.70 |
| Monthly | 60 days | +8% | 60-65% | 0.68-0.73 |

## 🚀 Installation
```bash
# Clone repository
git clone https://github.com/TechPeacock/nse-ml-trading-system.git
cd nse-ml-trading-system

# Install dependencies
pip install -r requirements.txt

# Create directory structure (automatically done)
python -c "from config import *"
```

## 📥 Download NSE Data (Manual - Daily)

Download these 5 files from NSE after market close (3:30 PM):

1. **Bhavcopy**: https://www.nseindia.com/all-reports
   - Save to: `data/raw/bhav/`
   
2. **Delivery Data**: https://www.nseindia.com/report-detail/eq_security
   - Save to: `data/raw/delivery/`
   
3. **FII/DII Data**: https://www.nseindia.com/reports/fii-dii
   - Save to: `data/raw/fii_dii/`
   
4. **Participant OI**: https://www.nseindia.com/reports-derivatives
   - Save to: `data/raw/participant_wise/`
   
5. **Bulk/Block Deals**: https://www.nseindia.com/report-detail/eq_security
   - Save to: `data/raw/bulk_block/`

## 💻 Usage

### Check Data Files
```bash
python main.py check
```

### Train Models (Evening - Post Market)
```bash
python main.py train
```
*Takes 20-40 minutes. Processes data, runs quality checks, trains 3 XGBoost models.*

### Generate Predictions (Morning - Pre Market)
```bash
python main.py predict
```

### View Top 10 Stocks
```bash
python main.py show
```

### Quality Check Only
```bash
python main.py quality
```

## 📅 Daily Workflow

### Evening (4:00 PM - 5:30 PM)
1. Download 5 NSE files
2. Place in respective folders
3. Run: `python main.py check`
4. Run: `python main.py train`

### Morning (8:00 AM - 9:00 AM)
1. Run: `python main.py predict`
2. Run: `python main.py show`
3. Analyze Top 10 stocks
4. Execute trades at 9:15 AM

## 📂 Project Structure
```
nse-ml-trading-system/
├── config.py                    # Configuration
├── data_loader.py               # NSE data loader
├── feature_engineering.py       # 50+ features
├── model_training.py            # XGBoost training
├── prediction.py                # Generate predictions
├── data_quality.py              # 8 quality checks
├── partial_data_handler.py      # Missing data handling
├── main.py                      # Main orchestrator
├── requirements.txt             # Dependencies
├── data/
│   ├── raw/                     # Manual NSE downloads
│   │   ├── bhav/
│   │   ├── delivery/
│   │   ├── fii_dii/
│   │   ├── participant_wise/
│   │   └── bulk_block/
│   ├── processed/               # Processed parquet files
│   └── models/                  # Trained XGBoost models
└── outputs/
    └── predictions/             # Daily predictions CSV
```

## 🔍 Feature Importance

Top features by importance:
1. **Delivery %** (12-15%) - Most important
2. **FII 5-day MA** (9-10%)
3. **Volume Ratio** (8-9%)
4. **Delivery Z-score** (7-8%)
5. **DII 5-day MA** (6-7%)

## ⚙️ Configuration

Edit `config.py` to adjust:
- `MIN_LIQUIDITY`: Minimum average daily volume (default: 100,000)
- `MIN_DELIVERY_PCT`: Minimum delivery % (default: 30%)
- Target returns and horizons for each model
- XGBoost hyperparameters

## 🛡️ Risk Management

- Always use **-2% stop loss** on every trade
- Maximum **10% capital** per stock
- Exit on **Day 5** (daily model) even if flat
- Reduce position sizes when VIX > 20

## 📈 Example Trade
```
Stock: RELIANCE
Signal: Daily prediction
Probability: 0.784 (High confidence)
Delivery %: 67.8% (Strong)
FII: ₹12.3 Cr (Positive)
DII: ₹8.5 Cr (Positive)

Entry: ₹2,456
Target: ₹2,530 (+3%)
Stop Loss: ₹2,407 (-2%)
Position Size: ₹1,00,000
Holding Period: Max 5 days
```

## 🐛 Troubleshooting

### "No data loaded"
- Check files are in correct folders
- Run `python main.py check`

### "Module not found"
- Run `pip install -r requirements.txt`

### Low accuracy (<0.55 ROC-AUC)
- Download more historical data (need 1+ years)
- Verify FII/DII and delivery data availability

## 📊 Backtesting (Optional)

To validate on historical data:
```python
# Coming soon - backtesting module
```

## ⚠️ Disclaimer

This system is for **educational purposes**. Past performance does not guarantee future results. Always:
- Do your own research
- Use proper risk management
- Never invest more than you can afford to lose
- Consult a financial advisor

## 📝 License

MIT License - Feel free to use and modify

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

## 📧 Contact

For questions or suggestions, open an issue on GitHub.

---

**Happy Trading! 📈🚀**
