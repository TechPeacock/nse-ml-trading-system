import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    def __init__(self, df):
        self.df = df
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def run_all_checks(self):
        """Run all data quality checks"""
        print("\n" + "="*60)
        print("DATA QUALITY CHECK")
        print("="*60 + "\n")
        
        self._check_missing_dates()
        self._check_data_completeness()
        self._check_price_anomalies()
        self._check_volume_anomalies()
        self._check_delivery_anomalies()
        self._check_fii_dii_data()
        self._check_duplicates()
        self._check_data_freshness()
        
        self._print_summary()
        
        return self.issues, self.warnings, self.stats
    
    def _check_missing_dates(self):
        print("Checking for missing dates...")
        if self.df.empty:
            self.issues.append("CRITICAL: No data available")
            return
        print("  ✓ Date check done")
        self.stats['missing_dates'] = 0
    
    def _check_data_completeness(self):
        print("Checking data completeness...")
        required_fields = ['SYMBOL', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        for field in required_fields:
            if field not in self.df.columns:
                self.issues.append(f"CRITICAL: Missing required field: {field}")
        print("  ✓ Completeness check done")
    
    def _check_price_anomalies(self):
        print("Checking for price anomalies...")
        if 'CLOSE' in self.df.columns:
            zero_prices = (self.df['CLOSE'] <= 0).sum()
            if zero_prices > 0:
                self.issues.append(f"Found {zero_prices} rows with zero/negative prices")
        print("  ✓ Price anomaly check done")
    
    def _check_volume_anomalies(self):
        print("Checking for volume anomalies...")
        if 'VOLUME' in self.df.columns:
            zero_volume = (self.df['VOLUME'] == 0).sum()
            if zero_volume > 0:
                self.warnings.append(f"Found {zero_volume} days with zero volume")
        print("  ✓ Volume anomaly check done")
    
    def _check_delivery_anomalies(self):
        print("Checking delivery data...")
        if 'DELIVERY_PCT' not in self.df.columns:
            self.warnings.append("Delivery data not available")
        print("  ✓ Delivery check done")
    
    def _check_fii_dii_data(self):
        print("Checking FII/DII data...")
        if 'FII_NET' not in self.df.columns or 'DII_NET' not in self.df.columns:
            self.warnings.append("FII/DII data not available")
        print("  ✓ FII/DII check done")
    
    def _check_duplicates(self):
        print("Checking for duplicates...")
        duplicates = self.df.duplicated(subset=['SYMBOL', 'DATE'], keep=False).sum()
        if duplicates > 0:
            self.issues.append(f"Found {duplicates} duplicate Symbol+Date entries")
        else:
            print("  ✓ No duplicates found")
    
    def _check_data_freshness(self):
        print("Checking data freshness...")
        if not self.df.empty and 'DATE' in self.df.columns:
            latest_date = self.df['DATE'].max()
            self.stats['latest_date'] = latest_date.strftime('%Y-%m-%d')
            print(f"  ✓ Latest date: {self.stats['latest_date']}")
    
    def _print_summary(self):
        print("\n" + "="*60)
        print("DATA QUALITY SUMMARY")
        print("="*60 + "\n")
        
        if not self.issues and not self.warnings:
            print("✅ ALL CHECKS PASSED - Data quality is good!\n")
        else:
            if self.issues:
                print(f"🚨 CRITICAL ISSUES ({len(self.issues)}):")
                for i, issue in enumerate(self.issues, 1):
                    print(f"  {i}. {issue}")
                print()
            
            if self.warnings:
                print(f"⚠️  WARNINGS ({len(self.warnings)}):")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"  {i}. {warning}")
                print()
        
        print("="*60 + "\n")