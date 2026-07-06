"""
Task 1: Complete EDA - Fixed Path Version
GMF Investments - Portfolio Optimization Challenge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import yfinance as yf

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP PATHS - FIX FOR FILE SAVING
# ============================================================================

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Create the directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"✅ Directory exists: {os.path.exists(DATA_DIR)}")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("✅ Libraries imported successfully!")

# ============================================================================
# DATA EXTRACTION
# ============================================================================

print("\n📊 Fetching data from YFinance...")
start = "2015-01-01"
end = "2026-06-30"
tickers = ["TSLA", "BND", "SPY"]

data_dict = {}
for ticker in tickers:
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    data_dict[ticker] = df
    print(f"    ✅ {len(df)} rows")

# Create adj_close DataFrame
adj_close = pd.DataFrame()
for ticker in tickers:
    if not data_dict[ticker].empty:
        if 'Close' in data_dict[ticker].columns:
            adj_close[ticker] = data_dict[ticker]['Close']
        elif 'Adj_Close' in data_dict[ticker].columns:
            adj_close[ticker] = data_dict[ticker]['Adj_Close']

print(f"\n✅ Data shape: {adj_close.shape}")
print(f"📅 Date range: {adj_close.index.min()} to {adj_close.index.max()}")

# ============================================================================
# DATA CLEANING
# ============================================================================

print("\n" + "="*60)
print("🧹 DATA CLEANING")
print("="*60)

print("Missing values before cleaning:")
print(adj_close.isnull().sum())

df_clean = adj_close.copy()
df_clean = df_clean.ffill().bfill()

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# ============================================================================
# SAVE CLEANED DATA
# ============================================================================

clean_data_path = os.path.join(DATA_DIR, 'adj_close_prices_clean.csv')
df_clean.to_csv(clean_data_path)
print(f"\n💾 Cleaned data saved to: {clean_data_path}")

# ============================================================================
# CALCULATE RETURNS
# ============================================================================

returns = df_clean.pct_change().dropna()
log_returns = np.log(df_clean / df_clean.shift(1)).dropna()

# Save returns
returns_path = os.path.join(DATA_DIR, 'returns_data.csv')
returns.to_csv(returns_path)
print(f"💾 Returns data saved to: {returns_path}")

print("\n" + "="*60)
print("📊 STATISTICAL SUMMARY")
print("="*60)
print(returns.describe())

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

print("\n" + "="*60)
print("📈 CREATING VISUALIZATIONS")
print("="*60)

# 1. Closing Prices
print("1. Closing Prices...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
for idx, (ticker, color) in enumerate(zip(df_clean.columns, colors)):
    ax = axes[idx]
    ax.plot(df_clean.index, df_clean[ticker], linewidth=2, color=color)
    ax.set_title(f'{ticker} - Adjusted Close Price', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
closing_prices_path = os.path.join(DATA_DIR, 'closing_prices.png')
plt.savefig(closing_prices_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {closing_prices_path}")

# 2. Returns Distribution
print("2. Returns Distribution...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, (ticker, color) in enumerate(zip(returns.columns, colors)):
    ax = axes[idx]
    ax.hist(returns[ticker], bins=50, alpha=0.7, color=color, edgecolor='black', density=True)
    ax.set_title(f'{ticker} - Daily Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Density')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=returns[ticker].mean(), color='red', linestyle='--', label=f'Mean: {returns[ticker].mean():.4f}')
    ax.legend()
plt.tight_layout()
returns_dist_path = os.path.join(DATA_DIR, 'returns_distribution.png')
plt.savefig(returns_dist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {returns_dist_path}")

# 3. Rolling Volatility
print("3. Rolling Volatility...")
window = 20
rolling_vol = pd.DataFrame()
for ticker in returns.columns:
    rolling_vol[ticker] = returns[ticker].rolling(window=window).std() * np.sqrt(252)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
for idx, (ticker, color) in enumerate(zip(returns.columns, colors)):
    ax = axes[idx]
    ax.plot(rolling_vol.index, rolling_vol[ticker], linewidth=2, color=color)
    ax.fill_between(rolling_vol.index, 0, rolling_vol[ticker], alpha=0.2, color=color)
    ax.set_title(f'{ticker} - Rolling Volatility (20-day)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.axhline(y=rolling_vol[ticker].mean(), color='red', linestyle='--', 
               label=f'Mean: {rolling_vol[ticker].mean():.2%}')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
rolling_vol_path = os.path.join(DATA_DIR, 'rolling_volatility.png')
plt.savefig(rolling_vol_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {rolling_vol_path}")

# 4. Correlation Heatmap
print("4. Correlation Heatmap...")
correlation_matrix = returns.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
corr_heatmap_path = os.path.join(DATA_DIR, 'correlation_heatmap.png')
plt.savefig(corr_heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {corr_heatmap_path}")

# 5. Boxplot of Returns
print("5. Returns Boxplot...")
fig, ax = plt.subplots(figsize=(12, 6))
returns_box = returns.copy()
returns_box.columns = [f'{col}\n(Vol: {returns[col].std():.2%})' for col in returns_box.columns]
returns_box.boxplot(ax=ax)
ax.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
ax.set_ylabel('Daily Return')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
boxplot_path = os.path.join(DATA_DIR, 'returns_boxplot.png')
plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {boxplot_path}")

# ============================================================================
# RISK METRICS
# ============================================================================

print("\n" + "="*60)
print("📊 RISK METRICS")
print("="*60)

def calculate_var(returns_series, confidence=0.95):
    return returns_series.quantile(1 - confidence)

def calculate_cvar(returns_series, confidence=0.95):
    var = calculate_var(returns_series, confidence)
    return returns_series[returns_series <= var].mean()

def calculate_sharpe(returns_series, risk_free=0.02):
    excess = returns_series.mean() - risk_free / 252
    return (excess * np.sqrt(252)) / returns_series.std()

risk_metrics = {}
for ticker in returns.columns:
    daily_returns = returns[ticker]
    risk_metrics[ticker] = {
        'Annualized Return': daily_returns.mean() * 252,
        'Annualized Volatility': daily_returns.std() * np.sqrt(252),
        '95% VaR': calculate_var(daily_returns, 0.95),
        '99% VaR': calculate_var(daily_returns, 0.99),
        '95% CVaR': calculate_cvar(daily_returns, 0.95),
        'Sharpe Ratio': calculate_sharpe(daily_returns),
    }

risk_df = pd.DataFrame(risk_metrics).T.round(4)
print(risk_df)

# Save risk metrics
risk_metrics_path = os.path.join(DATA_DIR, 'risk_metrics.csv')
risk_df.to_csv(risk_metrics_path)
print(f"\n💾 Risk metrics saved to: {risk_metrics_path}")

# Save correlation matrix
corr_matrix_path = os.path.join(DATA_DIR, 'correlation_matrix.csv')
correlation_matrix.to_csv(corr_matrix_path)
print(f"💾 Correlation matrix saved to: {corr_matrix_path}")

# ============================================================================
# STATIONARITY TESTING
# ============================================================================

print("\n" + "="*60)
print("📊 STATIONARITY TESTING (ADF Test)")
print("="*60)

from statsmodels.tsa.stattools import adfuller

def adf_test(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.6f}")
    is_stationary = result[1] < 0.05
    print(f"  ✅ Stationary: {is_stationary}")
    return is_stationary

# Test closing prices
print("\n🔍 Testing Closing Prices:")
stationarity_results = {}
for ticker in df_clean.columns:
    stationarity_results[f"{ticker}_price"] = adf_test(df_clean[ticker], f"{ticker} - Closing Prices")

# Test returns
print("\n🔍 Testing Daily Returns:")
for ticker in returns.columns:
    stationarity_results[f"{ticker}_returns"] = adf_test(returns[ticker], f"{ticker} - Returns")

# Save stationarity results
stationarity_df = pd.DataFrame({
    'Asset': list(stationarity_results.keys()),
    'Is_Stationary': [v for v in stationarity_results.values()]
})
stationarity_path = os.path.join(DATA_DIR, 'stationarity_summary.csv')
stationarity_df.to_csv(stationarity_path, index=False)
print(f"\n💾 Stationarity summary saved to: {stationarity_path}")

# ============================================================================
# COMPREHENSIVE SUMMARY
# ============================================================================

print("\n" + "="*60)
print("📊 COMPREHENSIVE SUMMARY")
print("="*60)

for ticker in returns.columns:
    print(f"\n🔹 {ticker}")
    print(f"  Price Range: ${df_clean[ticker].min():.2f} - ${df_clean[ticker].max():.2f}")
    print(f"  Mean Price: ${df_clean[ticker].mean():.2f}")
    print(f"  Mean Daily Return: {returns[ticker].mean():.4%}")
    print(f"  Daily Volatility: {returns[ticker].std():.4%}")
    print(f"  Annualized Return: {returns[ticker].mean() * 252:.4%}")
    print(f"  Annualized Volatility: {returns[ticker].std() * np.sqrt(252):.4%}")
    print(f"  Sharpe Ratio: {risk_metrics[ticker]['Sharpe Ratio']:.4f}")
    print(f"  95% VaR: {risk_metrics[ticker]['95% VaR']:.4%}")
    print(f"  99% VaR: {risk_metrics[ticker]['99% VaR']:.4%}")

# ============================================================================
# LIST ALL GENERATED FILES
# ============================================================================

print("\n" + "="*60)
print("📁 GENERATED FILES")
print("="*60)

print(f"\nDirectory: {DATA_DIR}")
for file in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file)
    size = os.path.getsize(file_path)
    print(f"  📄 {file} ({size:,} bytes)")

print("\n" + "="*60)
print("🎯 TASK 1 COMPLETED SUCCESSFULLY!")
print("="*60)