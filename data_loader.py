import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import zipfile
import warnings
warnings.filterwarnings('ignore')

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class NSEDataLoader:
    
    def __init__(self):
        self.bhav_dir = RAW_DATA_DIR / "bhav"
        self.delivery_dir = RAW_DATA_DIR / "delivery"
        self.fii_dii_dir = RAW_DATA_DIR / "fii_dii"
        self.participant_dir = RAW_DATA_DIR / "participant_wise"
        self.bulk_block_dir = RAW_DATA_DIR / "bulk_block"
        
        for dir_path in [self.bhav_dir, self.delivery_dir, self.fii_dii_dir, 
                         self.participant_dir, self.bulk_block_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_all_data(self):
        """Load and merge all data sources"""
        print("Loading NSE data from manual downloads...")
        
        df_bhav = self._load_bhavcopy()
        df_delivery = self._load_delivery()
        df_fii_dii = self._load_fii_dii()
        df_oi = self._load_participant_oi()
        df_bulk_block = self._load_bulk_block()
        
        print("\nMerging all data sources...")
        df_merged = self._merge_all_sources(
            df_bhav, df_delivery, df_fii_dii, df_oi, df_bulk_block
        )
        
        if not df_merged.empty:
            print(f"Final merged data: {len(df_merged)} rows, {df_merged['SYMBOL'].nunique()} unique symbols")
        
        return df_merged
    
    def _load_bhavcopy(self):
        """Load NEW NSE Bhavcopy format (2024+)"""
        print("\nLoading Bhavcopy files...")
        
        bhav_files = list(self.bhav_dir.glob("*.csv")) + list(self.bhav_dir.glob("*.zip"))
        
        if not bhav_files:
            print("⚠️  No bhavcopy files found!")
            return pd.DataFrame()
        
        all_data = []
        
        for file in bhav_files:
            try:
                # Handle ZIP files
                if file.suffix == '.zip':
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        csv_name = zip_ref.namelist()[0]
                        with zip_ref.open(csv_name) as csv_file:
                            df = pd.read_csv(csv_file)
                else:
                    df = pd.read_csv(file)
                
                # NEW NSE FORMAT - Map columns
                column_mapping = {
                    'TckrSymb': 'SYMBOL',
                    'TradDt': 'DATE',
                    'OpnPric': 'OPEN',
                    'HghPric': 'HIGH',
                    'LwPric': 'LOW',
                    'ClsPric': 'CLOSE',
                    'LastPric': 'LAST',
                    'TtlTradgVol': 'VOLUME',
                    'TtlNbOfTxsExctd': 'TRADES',
                    'SctySrs': 'SERIES'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Filter EQ series only
                if 'SERIES' in df.columns:
                    df = df[df['SERIES'] == 'EQ'].copy()
                
                # Use CLOSE if LAST is missing
                if 'CLOSE' not in df.columns and 'LAST' in df.columns:
                    df['CLOSE'] = df['LAST']
                
                # Check required columns
                required_cols = ['SYMBOL', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    all_data.append(df)
                    print(f"   ✓ Loaded {len(df)} rows from {file.name}")
                    
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not all_data:
            print("⚠️  No valid bhavcopy data loaded!")
            return pd.DataFrame()
        
        # Combine all data
        df_bhav = pd.concat(all_data, ignore_index=True)
        
        # Convert date and remove duplicates
        df_bhav['DATE'] = pd.to_datetime(df_bhav['DATE'], errors='coerce')
        df_bhav = df_bhav.dropna(subset=['DATE'])
        df_bhav = df_bhav.drop_duplicates(subset=['SYMBOL', 'DATE'], keep='first')
        df_bhav = df_bhav.sort_values(['SYMBOL', 'DATE']).reset_index(drop=True)
        
        print(f"   ✓ Total: {len(df_bhav)} rows, {df_bhav['SYMBOL'].nunique()} symbols")
        
        return df_bhav
    
    def _load_delivery(self):
        """Load NSE Delivery MTO format"""
        print("\nLoading Delivery data...")
        
        delivery_files = list(self.delivery_dir.glob("*.DAT")) + list(self.delivery_dir.glob("*.dat"))
        
        if not delivery_files:
            print("⚠️  No delivery files found!")
            return pd.DataFrame()
        
        all_data = []
        
        for file in delivery_files:
            try:
                # Read file
                with open(file, 'r') as f:
                    lines = f.readlines()
                
                # Find header row
                header_idx = None
                for i, line in enumerate(lines):
                    if 'Name of Security' in line:
                        header_idx = i
                        break
                
                if header_idx is None:
                    print(f"   ✗ Could not find header in {file.name}")
                    continue
                
                # Parse header and data
                header_line = lines[header_idx].strip()
                header = [h.strip() for h in header_line.split(',')]
                
                data_rows = []
                for line in lines[header_idx + 1:]:
                    line = line.strip()
                    if line.startswith('20,'):
                        row = [r.strip() for r in line.split(',')]
                        # Handle variable column counts
                        if len(row) < len(header):
                            row.extend([''] * (len(header) - len(row)))
                        elif len(row) > len(header):
                            row = row[:len(header)]
                        data_rows.append(row)
                
                if not data_rows:
                    print(f"   ✗ No data rows in {file.name}")
                    continue
                
                df = pd.DataFrame(data_rows, columns=header)
                
                # Map columns
                column_mapping = {
                    'Name of Security': 'SYMBOL_SERIES',
                    'Quantity Traded': 'TRADED_QTY',
                    'Deliverable Quantity(gross across client level)': 'DELIVERY_QTY',
                    '% of Deliverable Quantity to Traded Quantity': 'DELIVERY_PCT'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Extract SYMBOL (remove series suffix)
                if 'SYMBOL_SERIES' in df.columns:
                    df['SYMBOL'] = df['SYMBOL_SERIES'].str.strip()
                    df['SYMBOL'] = df['SYMBOL'].apply(lambda x: x[:-3].strip() if len(x) > 3 else x)
                
                # Convert to numeric
                for col in ['TRADED_QTY', 'DELIVERY_QTY', 'DELIVERY_PCT']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Extract date from filename
                date_str = self._extract_date_from_filename(file.name)
                if date_str:
                    df['DATE'] = pd.to_datetime(date_str, errors='coerce')
                
                if 'SYMBOL' in df.columns and 'DELIVERY_PCT' in df.columns:
                    all_data.append(df)
                    print(f"   ✓ Loaded {len(df)} rows from {file.name}")
                    
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not all_data:
            print("⚠️  No valid delivery data loaded!")
            return pd.DataFrame()
        
        df_delivery = pd.concat(all_data, ignore_index=True)
        df_delivery = df_delivery[df_delivery['SYMBOL'].notna()].copy()
        df_delivery = df_delivery.drop_duplicates(subset=['SYMBOL', 'DATE'], keep='first')
        
        print(f"   ✓ Total: {len(df_delivery)} rows")
        
        return df_delivery[['SYMBOL', 'DATE', 'DELIVERY_QTY', 'DELIVERY_PCT']].copy()
    
    def _load_fii_dii(self):
        """Load FII/DII data"""
        print("\nLoading FII/DII data...")
        
        fii_dii_files = list(self.fii_dii_dir.glob("*.csv")) + list(self.fii_dii_dir.glob("*.xlsx"))
        
        if not fii_dii_files:
            print("⚠️  No FII/DII files found!")
            return pd.DataFrame()
        
        all_data = []
        
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
                else:
                    df['DATE'] = pd.to_datetime('2025-10-24')
                
                all_data.append(df)
                
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not all_data:
            print("⚠️  No valid FII/DII data loaded!")
            return pd.DataFrame()
        
        df_fii_dii = pd.concat(all_data, ignore_index=True)
        print(f"   ✓ Loaded {len(df_fii_dii)} rows")
        
        return df_fii_dii
    
    def _load_participant_oi(self):
        """Load Participant-wise Open Interest"""
        print("\nLoading Participant OI data...")
        
        oi_files = list(self.participant_dir.glob("*.csv"))
        
        if not oi_files:
            return pd.DataFrame()
        
        all_data = []
        
        for file in oi_files:
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.upper().str.strip()
                
                date_str = self._extract_date_from_filename(file.name)
                if date_str:
                    df['DATE'] = pd.to_datetime(date_str, errors='coerce')
                
                all_data.append(df)
                
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df_oi = pd.concat(all_data, ignore_index=True)
        print(f"   ✓ Loaded {len(df_oi)} rows")
        
        return df_oi
    
    def _load_bulk_block(self):
        """Load Bulk/Block deal data"""
        print("\nLoading Bulk/Block deals...")
        
        deal_files = list(self.bulk_block_dir.glob("*.csv"))
        
        if not deal_files:
            return pd.DataFrame()
        
        all_data = []
        
        for file in deal_files:
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.upper().str.strip()
                
                date_str = self._extract_date_from_filename(file.name)
                if date_str:
                    df['DATE'] = pd.to_datetime(date_str, errors='coerce')
                else:
                    df['DATE'] = pd.to_datetime('2025-10-24')
                
                all_data.append(df)
                
            except Exception as e:
                print(f"   Error loading {file.name}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df_deals = pd.concat(all_data, ignore_index=True)
        print(f"   ✓ Loaded {len(df_deals)} rows")
        
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
            print(f"   Merged delivery data")
        else:
            df['DELIVERY_QTY'] = 0
            df['DELIVERY_PCT'] = 0
        
        # Add FII/DII (market-wide)
        if not df_fii_dii.empty:
            fii_dii_agg = df_fii_dii.groupby('DATE').first().reset_index()
            
            fii_cols = [c for c in fii_dii_agg.columns if 'FII' in c and any(x in c for x in ['NET', 'BUY'])]
            dii_cols = [c for c in fii_dii_agg.columns if 'DII' in c and any(x in c for x in ['NET', 'BUY'])]
            
            if fii_cols:
                fii_dii_agg['FII_NET'] = fii_dii_agg[fii_cols[0]]
            if dii_cols:
                fii_dii_agg['DII_NET'] = dii_dii_agg[dii_cols[0]]
            
            if 'FII_NET' in fii_dii_agg.columns and 'DII_NET' in fii_dii_agg.columns:
                df = df.merge(fii_dii_agg[['DATE', 'FII_NET', 'DII_NET']], on='DATE', how='left')
                print(f"   Merged FII/DII data")
        
        if 'FII_NET' not in df.columns:
            df['FII_NET'] = 0
        if 'DII_NET' not in df.columns:
            df['DII_NET'] = 0
        
        # Add OI
        if not df_oi.empty:
            oi_cols = [c for c in df_oi.columns if 'OI' in c.upper() and 'SYMBOL' not in c.upper()]
            if oi_cols and 'SYMBOL' in df_oi.columns:
                oi_by_symbol = df_oi.groupby(['SYMBOL', 'DATE'])[oi_cols[0]].sum().reset_index()
                oi_by_symbol.columns = ['SYMBOL', 'DATE', 'OI']
                df = df.merge(oi_by_symbol, on=['SYMBOL', 'DATE'], how='left')
        
        if 'OI' not in df.columns:
            df['OI'] = 0
        
        # Flag bulk/block deals
        if not df_bulk_block.empty and 'SYMBOL' in df_bulk_block.columns:
            bulk_deals = df_bulk_block[['SYMBOL', 'DATE']].drop_duplicates()
            bulk_deals['BULK_DEAL_FLAG'] = 1
            df = df.merge(bulk_deals, on=['SYMBOL', 'DATE'], how='left')
            df['BULK_DEAL_FLAG'] = df['BULK_DEAL_FLAG'].fillna(0)
            print(f"   Merged bulk/block deals")
        else:
            df['BULK_DEAL_FLAG'] = 0
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _extract_date_from_filename(self, filename):
        """Extract date from NSE filename"""
        import re
        
        # Pattern: 24102025 (DDMMYYYY)
        pattern1 = r'(\d{2})(\d{2})(\d{4})'
        match = re.search(pattern1, filename)
        if match:
            day, month, year = match.groups()
            try:
                return pd.to_datetime(f"{year}-{month}-{day}", format='%Y-%m-%d')
            except:
                pass
        
        return None
