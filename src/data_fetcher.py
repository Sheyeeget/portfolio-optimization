"""
Data fetching module for financial data from YFinance.
"""
import yfinance as yf
import pandas as pd
from typing import Dict
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and organize financial data from YFinance."""
    
    def __init__(self, start_date: str = "2015-01-01", end_date: str = "2026-06-30"):
        self.start_date = start_date
        self.end_date = end_date
        self.assets = {
            "TSLA": "Tesla Inc.",
            "BND": "Vanguard Total Bond Market ETF",
            "SPY": "S&P 500 ETF"
        }
        self.data = {}
    
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all assets."""
        for ticker in self.assets.keys():
            logger.info(f"Fetching data for {ticker}...")
            self.data[ticker] = self.fetch_single(ticker)
            time.sleep(0.5)  # Small delay to avoid rate limiting
        return self.data
    
    def fetch_single(self, ticker: str) -> pd.DataFrame:
        """Fetch data for a single asset."""
        try:
            # Using yf.download which is more reliable
            df = yf.download(
                ticker, 
                start=self.start_date, 
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [col.replace(' ', '_') for col in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # Ensure we have Adj_Close column
            if 'Adj_Close' not in df.columns and 'Close' in df.columns:
                df['Adj_Close'] = df['Close']
            elif 'Adj_Close' not in df.columns:
                df['Adj_Close'] = df['Close']
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def get_adj_close_pivot(self) -> pd.DataFrame:
        """Get Adjusted Close prices in pivot table format."""
        if not self.data:
            self.fetch_all()
        
        adj_close_dict = {}
        for ticker, df in self.data.items():
            if df is not None and not df.empty:
                # Check for Adj_Close column (with various possible names)
                if 'Adj_Close' in df.columns:
                    adj_close_dict[ticker] = df['Adj_Close']
                elif 'Close' in df.columns:
                    adj_close_dict[ticker] = df['Close']
                elif 'Adj_Close' not in df.columns and 'Close' in df.columns:
                    adj_close_dict[ticker] = df['Close']
                else:
                    # If no price column found, use the first column
                    logger.warning(f"No price column found for {ticker}, using first column")
                    adj_close_dict[ticker] = df.iloc[:, 0]
            else:
                logger.warning(f"No data available for {ticker}")
        
        if adj_close_dict:
            result = pd.DataFrame(adj_close_dict)
            logger.info(f"Created pivot table with shape: {result.shape}")
            return result
        else:
            logger.warning("No data available for any asset")
            return pd.DataFrame()