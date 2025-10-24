import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import zipfile
import warnings
warnings.filterwarnings('ignore')

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class NSEDataLoader:
    """
    Intelligently loads and merges NSE data from manually downloaded files.
    Handles various NSE file formats and naming conventions.
    """
    
    def __init__(self):
        self.bhav_dir = RAW_DATA_DIR / "bhav"
        self.delivery_dir = RAW_DATA_DIR / "delivery"
        self.fii_dii_dir = RAW_DATA_DIR / "fii_dii"
        self.participant_dir = RAW_DATA_DIR / "participant_wise"
        self.bulk_block_dir = RAW_DATA_DIR / "bulk_block"
        
        # Create directories if they don't exist
        for dir_path in [self.bhav_dir, self.delivery_dir, self.fii_dii_dir, 
                         self.participant_dir, self.bulk_block_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_all_data(self):
        """Load and merge all data sources"""
        print("Loading NSE data from manual downloads...")
        
        # Load each data source
        df_bhav = self._load_bhavcopy()
        df_delivery = self._load_delivery()
        df_fii_dii = self._load_fii_dii()
        df_oi = self._load_participant_oi()
        df_bulk_block = self._load_bulk_block()
        
        # Merge all data
        print("\nMerging all data sources...")
        df_merged = self._merge_all_sources(
            df_bhav, df_delivery, df_fii_dii, df_oi, df_bulk_block
        )
        
        if not df_merged.empty:
            print(f"Final merged data: {len(df_merged)} rows, {df_merged['SYMBOL'].nunique()} unique symbols")
        
        return df_merged
    
    def _load_bhavcopy(self):
        """Load Equity Bhavcopy (OHLCV data)"""
        print("\nLoading Bhavcopy files...")
        
        bhav_files = list(self.bhav_dir.glob("*.csv*")) + list(self.bhav_dir.glob("*.zip"))
        
        if not bhav_files:
            print("⚠️  No bhavcopy files found!")
            print(f"   Please download from NSE and place in: {self.bhav_dir}")
            return pd.DataFrame()
        
        df_list = []
        
        for file in bhav_files:
            try:
                # Handle zip files
                if file.suffix == '.zip':
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        csv_name = zip_ref.namelist()[0]
                        with zip_ref.open(csv_name) as csv_file:
                            df = pd.read_csv(csv_file)
                else:
                    df = pd.read_csv(file)
                
                # Standardize column names
                df.columns = df.columns.str.upper().str.strip()
                
                # Filter only EQ series
                if 'SERIES' in df.columns:
                    df = df[df['SERIES'] == 'EQ'].copy()
                
                # Rename columns
                column_mapping = {
                    'SYMBOL': 'SYMBOL',
                    'TIMESTAMP': 'DATE',
                    'OPEN': 'OPEN',
                    'HIGH': 'HIGH',
                    'LOW': 'LOW',
                    'CLOSE': 'CLOSE',
                    'LAST': 'CLOSE',
                    'PREVCLOSE': 'PREV_CLOSE',
                    'TOTTRDQTY': 'VOLUME',
                    'TOTTRDVAL': 'TURNOVER',
                    'TOTALTRADES': 'TRADES'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Ensure required columns exist
                required_cols = ['SYMBOL', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                if all(col in df.columns for col in required_cols):
                    df_list.append(df[required_cols])
                    
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not df_list:
            print("⚠️  No valid bhavcopy data loaded!")
            return pd.DataFrame()
        
        df_bhav = pd.concat(df_list, ignore_index=True)
        df_bhav['DATE'] = pd.to_datetime(df_bhav['DATE'], errors='coerce')
        df_bhav = df_bhav.dropna(subset=['DATE'])
        df_bhav = df_bhav.sort_values(['SYMBOL', 'DATE']).reset_index(drop=True)
        
        print(f"   ✓ Loaded {len(df_bhav)} rows from {len(bhav_files)} files")
        
        return df_bhav
    
    def _load_delivery(self):
        """Load Delivery Position data"""
        print("\nLoading Delivery data...")
        
        delivery_files = list(self.delivery_dir.glob("*"))
        
        if not delivery_files:
            print("⚠️  No delivery files found!")
            return pd.DataFrame()
        
        df_list = []
        
        for file in delivery_files:
            try:
                if file.suffix.lower() in ['.dat', '.csv', '.txt']:
                    # Try different delimiters
                    for delimiter in [',', '|', '\t']:
                        try:
                            df = pd.read_csv(file, delimiter=delimiter)
                            if len(df.columns) > 1:
                                break
                        except:
                            continue
                    
                    df.columns = df.columns.str.upper().str.strip()
                    
                    column_mapping = {
                        'SYMBOL': 'SYMBOL',
                        'SERIES': 'SERIES',
                        'TRADED QUANTITY': 'TRADED_QTY',
                        'DELIVERABLE QUANTITY': 'DELIVERY_QTY',
                        'DELIVERY QTY': 'DELIVERY_QTY',
                        '% DELI. QTY TO TRADED QTY': 'DELIVERY_PCT',
                        'DELIVERY PERCENTAGE': 'DELIVERY_PCT'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    if 'SERIES' in df.columns:
                        df = df[df['SERIES'] == 'EQ'].copy()
                    
                    date_str = self._extract_date_from_filename(file.name)
                    if date_str:
                        df['DATE'] = pd.to_datetime(date_str, errors='coerce')
                    
                    if 'SYMBOL' in df.columns and 'DELIVERY_QTY' in df.columns:
                        df_list.append(df)
                        
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not df_list:
            print("⚠️  No valid delivery data loaded!")
            return pd.DataFrame()
        
        df_delivery = pd.concat(df_list, ignore_index=True)
        
        if 'DELIVERY_PCT' not in df_delivery.columns and 'TRADED_QTY' in df_delivery.columns:
            df_delivery['DELIVERY_PCT'] = (
                df_delivery['DELIVERY_QTY'] / df_delivery['TRADED_QTY'] * 100
            ).fillna(0)
        
        print(f"   ✓ Loaded {len(df_delivery)} rows from {len(delivery_files)} files")
        
        return df_delivery[['SYMBOL', 'DATE', 'DELIVERY_QTY', 'DELIVERY_PCT']].copy()
    
    def _load_fii_dii(self):
        """Load FII/DII data"""
        print("\nLoading FII/DII data...")
        
        fii_dii_files = list(self.fii_dii_dir.glob("*.csv")) + list(self.fii_dii_dir.glob("*.xlsx"))
        
        if not fii_dii_files:
            print("⚠️  No FII/DII files found!")
            return pd.DataFrame()
        
        df_list = []
        
        for file in fii_dii_files:
            try:
                if file.suffix == '.xlsx':
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)
                    
                df.columns = df.columns.str.upper().str.strip()
                
                date_str = self._extract_date_from_filename(file.name)
                if date_str:
                    df['DATE'] = pd.to_datetime(date_str, errors='coerce')
                
                df_list.append(df)
                
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not df_list:
            print("⚠️  No valid FII/DII data loaded!")
            return pd.DataFrame()
        
        df_fii_dii = pd.concat(df_list, ignore_index=True)
        
        print(f"   ✓ Loaded {len(df_fii_dii)} rows from {len(fii_dii_files)} files")
        
        return df_fii_dii
    
    def _load_participant_oi(self):
        """Load Participant-wise Open Interest"""
        print("\nLoading Participant OI data...")
        
        oi_files = list(self.participant_dir.glob("*.csv"))
        
        if not oi_files:
            print("⚠️  No participant OI files found!")
            return pd.DataFrame()
        
        df_list = []
        
        for file in oi_files:
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.upper().str.strip()
                
                date_str = self._extract_date_from_filename(file.name)
                if date_str:
                    df['DATE'] = pd.to_datetime(date_str, errors='coerce')
                
                df_list.append(df)
                
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not df_list:
            return pd.DataFrame()
        
        df_oi = pd.concat(df_list, ignore_index=True)
        
        print(f"   ✓ Loaded {len(df_oi)} rows from {len(oi_files)} files")
        
        return df_oi
    
    def _load_bulk_block(self):
        """Load Bulk/Block deal data"""
        print("\nLoading Bulk/Block deals...")
        
        deal_files = list(self.bulk_block_dir.glob("*.csv"))
        
        if not deal_files:
            print("⚠️  No bulk/block deal files found!")
            return pd.DataFrame()
        
        df_list = []
        
        for file in deal_files:
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.upper().str.strip()
                
                date_str = self._extract_date_from_filename(file.name)
                if date_str:
                    df['DATE'] = pd.to_datetime(date_str, errors='coerce')
                
                df_list.append(df)
                
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not df_list:
            return pd.DataFrame()
        
        df_deals = pd.concat(df_list, ignore_index=True)
        
        print(f"   ✓ Loaded {len(df_deals)} rows from {len(deal_files)} files")
        
        return df_deals
    
    def _merge_all_sources(self, df_bhav, df_delivery, df_fii_dii, df_oi, df_bulk_block):
        """Merge all data sources"""
        
        if df_bhav.empty:
            print("ERROR: No bhavcopy data available!")
            return pd.DataFrame()
        
        df = df_bhav.copy()
        
        # Merge delivery
        if not df_delivery.empty:
            df = df.merge(df_delivery, on=['SYMBOL', 'DATE'], how='left')
        else:
            df['DELIVERY_QTY'] = 0
            df['DELIVERY_PCT'] = 0
        
        # Add FII/DII (market-wide)
        if not df_fii_dii.empty:
            fii_dii_agg = df_fii_dii.groupby('DATE').first().reset_index()
            
            fii_col = [c for c in fii_dii_agg.columns if 'FII' in c.upper() and 'NET' in c.upper()]
            dii_col = [c for c in fii_dii_agg.columns if 'DII' in c.upper() and 'NET' in c.upper()]
            
            if fii_col:
                fii_dii_agg = fii_dii_agg.rename(columns={fii_col[0]: 'FII_NET'})
            if dii_col:
                fii_dii_agg = fii_dii_agg.rename(columns={dii_col[0]: 'DII_NET'})
            
            if 'FII_NET' in fii_dii_agg.columns and 'DII_NET' in fii_dii_agg.columns:
                df = df.merge(fii_dii_agg[['DATE', 'FII_NET', 'DII_NET']], on='DATE', how='left')
        
        if 'FII_NET' not in df.columns:
            df['FII_NET'] = 0
        if 'DII_NET' not in df.columns:
            df['DII_NET'] = 0
        
        # Add OI
        if not df_oi.empty and 'OI' in df_oi.columns:
            oi_by_symbol = df_oi.groupby(['SYMBOL', 'DATE'])['OI'].sum().reset_index()
            df = df.merge(oi_by_symbol, on=['SYMBOL', 'DATE'], how='left')
        else:
            df['OI'] = 0
        
        # Flag bulk/block deals
        if not df_bulk_block.empty:
            bulk_deals = df_bulk_block[['SYMBOL', 'DATE']].drop_duplicates()
            bulk_deals['BULK_DEAL_FLAG'] = 1
            df = df.merge(bulk_deals, on=['SYMBOL', 'DATE'], how='left')
            df['BULK_DEAL_FLAG'] = df['BULK_DEAL_FLAG'].fillna(0)
        else:
            df['BULK_DEAL_FLAG'] = 0
        
        # Fill missing
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _extract_date_from_filename(self, filename):
        """Extract date from NSE filename"""
        import re
        
        # Pattern 1: cm25OCT2025
        pattern1 = r'(\d{2})([A-Z]{3})(\d{4})'
        match = re.search(pattern1, filename.upper())
        if match:
            day, month, year = match.groups()
            date_str = f"{day}{month}{year}"
            try:
                return pd.to_datetime(date_str, format='%d%b%Y')
            except:
                pass
        
        # Pattern 2: 25102025
        pattern2 = r'(\d{8})'
        match = re.search(pattern2, filename)
        if match:
            date_str = match.group(1)
            try:
                return pd.to_datetime(date_str, format='%d%m%Y')
            except:
                try:
                    return pd.to_datetime(date_str, format='%Y%m%d')
                except:
                    pass
        
        return None
