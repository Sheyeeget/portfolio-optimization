import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess financial data for modeling."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.processed_data = None
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the data by handling missing values."""
        df = self.data.copy()
        df = df.ffill().bfill()
        self.processed_data = df
        return df
    
    def calculate_returns(self, price_col: str = 'Adj_Close') -> pd.DataFrame:
        """Calculate daily returns."""
        df = self.processed_data.copy() if self.processed_data is not None else self.data.copy()
        df['Daily_Return'] = df[price_col].pct_change()
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        return df
    

def test_stationarity(series: pd.Series, significance_level: float = 0.05) -> dict:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(series.dropna(), autolag='AIC')
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level,
        'used_lag': result[2],
        'n_observations': result[3]
    }