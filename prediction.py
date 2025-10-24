import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PartialDataHandler:
    """Handles scenarios where daily downloads are missed"""
    
    def __init__(self, df):
        self.df = df
        self.missing_dates = []
        self.imputed_rows = []
    
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
        elif strategy == 'mark_only':
            return self._mark_missing_only()
        else:
            print(f"Unknown strategy: {strategy}")
            return self.df
    
    def _forward_fill(self):
        """Forward fill missing dates"""
        df = self.df.copy()
        
        symbols = df['SYMBOL'].unique()
        
        min_date = df['DATE'].min()
        max_date = df['DATE'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        all_dates = all_dates[all_dates.weekday < 5]
        
        complete_index = pd.MultiIndex.from_product(
            [symbols, all_dates],
            names=['SYMBOL', 'DATE']
        )
        
        df_filled = df.set_index(['SYMBOL', 'DATE']).reindex(complete_index)
        
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_cols] = df_filled.groupby('SYMBOL')[numeric_cols].ffill()
        
        df_filled = df_filled.reset_index()
        
        imputed_count = df_filled[numeric_cols[0]].isna().sum()
        print(f"  ✓ Forward filled {imputed_count:,} missing rows")
        
        return df_filled
    
    def _interpolate(self):
        """Interpolate missing values"""
        df = self.df.copy()
        
        df = df.sort_values(['SYMBOL', 'DATE'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df[col] = df.groupby('SYMBOL')[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        
        print("  ✓ Interpolated missing values")
        
        return df
    
    def _skip_incomplete_symbols(self):
        """Remove symbols with missing data"""
        df = self.df.copy()
        
        date_counts = df.groupby('SYMBOL')['DATE'].nunique()
        max_dates = date_counts.max()
        
        complete_symbols = date_counts[date_counts == max_dates].index
        df_complete = df[df['SYMBOL'].isin(complete_symbols)].copy()
        
        removed = len(df) - len(df_complete)
        print(f"  ✓ Removed {removed:,} rows from incomplete symbols")
        
        return df_complete
    
    def _mark_missing_only(self):
        """Just flag missing data"""
        df = self.df.copy()
        df['HAS_DATA'] = True
        
        print(f"  ✓ Marked missing dates")
        
        return df
    
    def print_coverage_summary(self):
        """Print summary of data coverage"""
        if self.df.empty:
            print("No data available for coverage report")
            return
        
        min_date = self.df['DATE'].min()
        max_date = self.df['DATE'].max()
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        expected_days = len(date_range[date_range.weekday < 5])
        
        coverage = self.df.groupby('SYMBOL').agg({
            'DATE': ['count', 'min', 'max']
        }).reset_index()
        
        coverage.columns = ['SYMBOL', 'Days_Available', 'First_Date', 'Last_Date']
        coverage['Expected_Days'] = expected_days
        coverage['Coverage_Pct'] = (coverage['Days_Available'] / expected_days * 100).round(2)
        
        print("\n" + "="*60)
        print("DATA COVERAGE SUMMARY")
        print("="*60 + "\n")
        
        print(f"Total Symbols: {len(coverage)}")
        print(f"Average Coverage: {coverage['Coverage_Pct'].mean():.1f}%")
        print(f"Symbols with 100% coverage: {(coverage['Coverage_Pct'] == 100).sum()}")
        print(f"Symbols with <90% coverage: {(coverage['Coverage_Pct'] < 90).sum()}")
        
        print("\n" + "="*60 + "\n")
