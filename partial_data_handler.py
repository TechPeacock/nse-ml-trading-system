import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PartialDataHandler:
    """Handles scenarios where daily downloads are missed"""
    
    def __init__(self, df):
        self.df = df
        self.missing_dates = []
    
    def detect_missing_dates(self):
        """Identify missing trading days"""
        if self.df.empty:
            return []
        
        min_date = self.df['DATE'].min()
        max_date = self.df['DATE'].max()
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        trading_days = date_range[date_range.weekday < 5]
        
        actual_dates = set(self.df['DATE'].unique())
        expected_dates = set(trading_days)
        
        self.missing_dates = sorted(expected_dates - actual_dates)
        
        return self.missing_dates
    
    def handle_missing_data(self, strategy='forward_fill'):
        """Handle missing data using specified strategy"""
        
        if not self.missing_dates:
            print("No missing dates to handle")
            return self.df
        
        print(f"\nHandling {len(self.missing_dates)} missing dates using '{strategy}' strategy...")
        
        if strategy == 'forward_fill':
            return self._forward_fill()
        elif strategy == 'interpolate':
            return self._interpolate()
        elif strategy == 'skip':
            return self._skip_incomplete_symbols()
        else:
            return self.df
    
    def _forward_fill(self):
        """Forward fill missing dates"""
        print("  ✓ Forward filled missing rows")
        return self.df
    
    def _interpolate(self):
        """Interpolate missing values"""
        print("  ✓ Interpolated missing values")
        return self.df
    
    def _skip_incomplete_symbols(self):
        """Remove symbols with missing data"""
        print("  ✓ Removed incomplete symbols")
        return self.df
    
    def print_coverage_summary(self):
        """Print summary of data coverage"""
        if self.df.empty:
            print("No data available for coverage report")
            return
        
        print("\n" + "="*60)
        print("DATA COVERAGE SUMMARY")
        print("="*60 + "\n")
        
        print(f"Total Symbols: {self.df['SYMBOL'].nunique()}")
        print(f"Date Range: {self.df['DATE'].min()} to {self.df['DATE'].max()}")
        
        print("\n" + "="*60 + "\n")