import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class FeatureEngine:

    def __init__(self, df):
        """df must have: DATE, SYMBOL, OPEN, HIGH, LOW, CLOSE, VOLUME, DELIVERY_PCT, FII_NET, DII_NET, OI"""
        self.df = df.sort_values(['SYMBOL', 'DATE']).reset_index(drop=True)

    def create_all_features(self):
        """Master feature engineering pipeline"""
        print("Creating price/volume features...")
        self.df = self._price_volume_features(self.df)

        print("Creating smart money features...")
        self.df = self._smart_money_features(self.df)

        print("Creating FII/DII features...")
        self.df = self._fii_dii_features(self.df)

        print("Creating liquidity features...")
        self.df = self._liquidity_features(self.df)

        print("Creating sector features...")
        self.df = self._sector_features(self.df)

        print("Creating labels...")
        self.df = self._create_labels(self.df)

        return self.df

    def _price_volume_features(self, df):
        """Price and volume indicators"""
        df = df.copy()

        # Returns
        df['returns_1d'] = df.groupby('SYMBOL')['CLOSE'].pct_change()
        df['returns_5d'] = df.groupby('SYMBOL')['CLOSE'].pct_change(5)
        df['returns_20d'] = df.groupby('SYMBOL')['CLOSE'].pct_change(20)

        # Volume
        df['log_volume'] = np.log1p(df['VOLUME'])
        df['volume_ma5'] = df.groupby('SYMBOL')['VOLUME'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['volume_ma20'] = df.groupby('SYMBOL')['VOLUME'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['volume_ma5_ratio'] = df['VOLUME'] / (df['volume_ma5'] + 1)
        df['volume_ma20_ratio'] = df['VOLUME'] / (df['volume_ma20'] + 1)

        # Range
        df['high_low_range'] = (df['HIGH'] - df['LOW']) / (df['CLOSE'] + 1)

        # VWAP
        df['vwap'] = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3
        df['vwap_ma5'] = df.groupby('SYMBOL')['vwap'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['vwap_deviation'] = (df['CLOSE'] - df['vwap_ma5']) / (df['vwap_ma5'] + 1)

        # OBV
        df['obv'] = df.groupby('SYMBOL').apply(
            lambda x: (np.sign(x['returns_1d'].fillna(0)) * x['VOLUME']).cumsum()
        ).reset_index(level=0, drop=True)
        df['obv_norm'] = df.groupby('SYMBOL')['obv'].transform(
            lambda x: (x - x.rolling(20, min_periods=1).mean()) / (x.rolling(20, min_periods=1).std() + 1)
        )

        # NR7
        df['range'] = df['HIGH'] - df['LOW']
        df['nr7_flag'] = (df['range'] == df.groupby('SYMBOL')['range'].transform(
            lambda x: x.rolling(7, min_periods=1).min()
        )).astype(int)

        # ATR
        prev_close = df.groupby('SYMBOL')['CLOSE'].shift(1)
        df['tr'] = np.maximum(
            df['HIGH'] - df['LOW'],
            np.maximum(
                abs(df['HIGH'] - prev_close),
                abs(df['LOW'] - prev_close)
            )
        )
        df['atr'] = df.groupby('SYMBOL')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df['atr_norm'] = df['atr'] / (df['CLOSE'] + 1)

        # Bollinger Bands
        df['bb_mid'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['bb_std'] = df.groupby('SYMBOL')['CLOSE'].transform(lambda x: x.rolling(20, min_periods=1).std())
        df['bb_width_norm'] = (2 * df['bb_std']) / (df['bb_mid'] + 1)

        return df

    def _smart_money_features(self, df):
        """Delivery, OI features"""
        df = df.copy()

        # Delivery features
        df['delivery_pct'] = df['DELIVERY_PCT'].fillna(0)
        df['delivery_zscore_20d'] = df.groupby('SYMBOL')['delivery_pct'].transform(
            lambda x: (x - x.rolling(20, min_periods=1).mean()) / (x.rolling(20, min_periods=1).std() + 1)
        )
        df['delivery_trend_5d'] = df.groupby('SYMBOL')['delivery_pct'].transform(lambda x: x.diff(5))

        # OI
        df['oi_change_pct'] = df.groupby('SYMBOL')['OI'].pct_change().fillna(0)
        df['oi_delta_5d'] = df.groupby('SYMBOL')['OI'].transform(
            lambda x: x.diff(5) / (x.shift(5) + 1)
        ).fillna(0)

        # Flags
        df['bulk_deal_flag'] = df.get('BULK_DEAL_FLAG', pd.Series([0] * len(df))).fillna(0)
        df['block_deal_flag'] = 0
        df['pcr_ratio'] = 0
        df['max_pain_distance'] = 0

        return df

    def _fii_dii_features(self, df):
        """FII/DII flows"""
        df = df.copy()

        df['fii_net_delta'] = df.get('FII_NET', pd.Series([0] * len(df))).fillna(0)
        df['fii_net_ma5'] = df.groupby('SYMBOL')['fii_net_delta'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df['fii_net_ma20'] = df.groupby('SYMBOL')['fii_net_delta'].transform(
            lambda x: x.rolling(20, min_periods=1).mean())

        df['dii_net_delta'] = df.get('DII_NET', pd.Series([0] * len(df))).fillna(0)
        df['dii_net_ma5'] = df.groupby('SYMBOL')['dii_net_delta'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df['dii_net_ma20'] = df.groupby('SYMBOL')['dii_net_delta'].transform(
            lambda x: x.rolling(20, min_periods=1).mean())

        df['fii_dii_divergence'] = df['fii_net_ma5'] - df['dii_net_ma5']
        df['institutional_flow_strength'] = (df['fii_net_ma5'] + df['dii_net_ma5']) / (
                    df['volume_ma5'] * df['CLOSE'] + 1)

        return df

    def _liquidity_features(self, df):
        """Liquidity metrics"""
        df = df.copy()

        df['avg_volume_20d'] = df.groupby('SYMBOL')['VOLUME'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['spread_proxy'] = (df['HIGH'] - df['LOW']) / (df['CLOSE'] + 1)
        df['market_cap_log'] = np.log1p(df['VOLUME'] * df['CLOSE'])
        df['free_float_pct'] = 50

        return df

    def _sector_features(self, df):
        """Sector relative strength"""
        df = df.copy()

        df['sector_relative_strength_5d'] = 0
        df['sector_relative_strength_20d'] = 0

        return df

    def _create_labels(self, df):
        """Create forward-looking labels"""
        from config import DAILY_LABEL, WEEKLY_LABEL, MONTHLY_LABEL

        df = df.copy()

        # Daily
        df['future_return_5d'] = df.groupby('SYMBOL')['CLOSE'].transform(
            lambda x: x.shift(-DAILY_LABEL['horizon']) / x - 1
        )
        df['label_daily'] = (df['future_return_5d'] >= DAILY_LABEL['return_threshold']).astype(int)

        # Weekly
        df['future_return_20d'] = df.groupby('SYMBOL')['CLOSE'].transform(
            lambda x: x.shift(-WEEKLY_LABEL['horizon']) / x - 1
        )
        df['label_weekly'] = (df['future_return_20d'] >= WEEKLY_LABEL['return_threshold']).astype(int)

        # Monthly
        df['future_return_60d'] = df.groupby('SYMBOL')['CLOSE'].transform(
            lambda x: x.shift(-MONTHLY_LABEL['horizon']) / x - 1
        )
        df['label_monthly'] = (df['future_return_60d'] >= MONTHLY_LABEL['return_threshold']).astype(int)

        return df