"""
Task 4: Portfolio Optimization Using Modern Portfolio Theory (MPT) - FAST VERSION
GMF Investments - Portfolio Optimization Challenge
Goal: Build optimal portfolio using Efficient Frontier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(DATA_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("🚀 TASK 4: PORTFOLIO OPTIMIZATION USING MPT (FAST)")
print("="*60)
print(f"📁 Data directory: {DATA_DIR}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 1: LOAD DATA")
print("="*60)

# Load cleaned data
data_path = os.path.join(DATA_DIR, 'adj_close_prices_clean.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"✅ Loaded {len(df)} rows of data")

# Calculate historical returns
returns = df.pct_change().dropna()
print(f"✅ Returns shape: {returns.shape}")

# ============================================================================
# STEP 2: PREPARE EXPECTED RETURNS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 2: PREPARE EXPECTED RETURNS")
print("="*60)

# Load future forecast for TSLA (from Task 3)
forecast_path = os.path.join(DATA_DIR, 'future_forecast_12months.csv')
future_forecast = pd.read_csv(forecast_path, index_col=0, parse_dates=True)

# Calculate expected return for TSLA from forecast
tsla_forecast_return = (future_forecast.iloc[-1] / future_forecast.iloc[0]) - 1

# Convert to scalar if it's a Series
if hasattr(tsla_forecast_return, 'iloc'):
    tsla_forecast_return = tsla_forecast_return.iloc[0]
elif hasattr(tsla_forecast_return, 'values'):
    tsla_forecast_return = tsla_forecast_return.values[0]

tsla_annualized_return = (1 + tsla_forecast_return) ** (252 / len(future_forecast)) - 1

print(f"TSLA Forecast Return (12 months): {tsla_forecast_return:.2%}")
print(f"TSLA Annualized Return: {tsla_annualized_return:.2%}")

# Calculate historical returns for BND and SPY
bnd_historical_return = returns['BND'].mean() * 252
spy_historical_return = returns['SPY'].mean() * 252

print(f"BND Historical Annualized Return: {bnd_historical_return:.2%}")
print(f"SPY Historical Annualized Return: {spy_historical_return:.2%}")

# Create expected returns vector
expected_returns = {
    'TSLA': tsla_annualized_return,
    'BND': bnd_historical_return,
    'SPY': spy_historical_return
}

expected_returns_array = np.array(list(expected_returns.values()))
tickers = list(expected_returns.keys())

expected_returns_df = pd.DataFrame(list(expected_returns.values()), 
                                   index=list(expected_returns.keys()),
                                   columns=['Expected Return'])
print("\n📊 Expected Returns:")
print(expected_returns_df)

# ============================================================================
# STEP 3: COMPUTE COVARIANCE MATRIX
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 3: COMPUTE COVARIANCE MATRIX")
print("="*60)

# Calculate covariance matrix (annualized)
cov_matrix = returns.cov() * 252
print("\n📊 Annualized Covariance Matrix:")
print(cov_matrix)

# Visualize covariance matrix as heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cov_matrix, annot=True, fmt='.4f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Asset Covariance Matrix (Annualized)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'covariance_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Covariance heatmap saved")

# ============================================================================
# STEP 4: GENERATE EFFICIENT FRONTIER (FAST METHOD)
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 4: GENERATE EFFICIENT FRONTIER (FAST)")
print("="*60)

print("📈 Generating Efficient Frontier using Monte Carlo simulation...")

# Use Monte Carlo simulation to generate random portfolios
n_portfolios = 10000
np.random.seed(42)

# Store results
portfolio_returns = []
portfolio_risks = []
portfolio_sharpe = []
portfolio_weights = []

risk_free_rate = 0.02

for i in range(n_portfolios):
    # Generate random weights
    weights = np.random.random(len(tickers))
    weights = weights / np.sum(weights)
    
    # Calculate portfolio return
    ret = np.sum(weights * expected_returns_array)
    
    # Calculate portfolio risk
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
    
    # Calculate Sharpe ratio
    sharpe = (ret - risk_free_rate) / risk if risk > 0 else 0
    
    portfolio_returns.append(ret)
    portfolio_risks.append(risk)
    portfolio_sharpe.append(sharpe)
    portfolio_weights.append(weights)

print(f"✅ Generated {n_portfolios} random portfolios")

# ============================================================================
# STEP 5: FIND OPTIMAL PORTFOLIOS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 5: FIND OPTIMAL PORTFOLIOS")
print("="*60)

# Convert to numpy arrays
portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
portfolio_sharpe = np.array(portfolio_sharpe)
portfolio_weights = np.array(portfolio_weights)

# Find Maximum Sharpe Ratio portfolio
max_sharpe_idx = np.argmax(portfolio_sharpe)
weights_max_sharpe = portfolio_weights[max_sharpe_idx]
ret_max_sharpe = portfolio_returns[max_sharpe_idx]
risk_max_sharpe = portfolio_risks[max_sharpe_idx]
sharpe_max_sharpe = portfolio_sharpe[max_sharpe_idx]

print(f"\n📈 Maximum Sharpe Ratio Portfolio:")
print(f"  Expected Return: {ret_max_sharpe:.2%}")
print(f"  Expected Volatility: {risk_max_sharpe:.2%}")
print(f"  Sharpe Ratio: {sharpe_max_sharpe:.2f}")
print("  Weights:")
for ticker, weight in zip(tickers, weights_max_sharpe):
    print(f"    {ticker}: {weight:.2%}")

# Find Minimum Volatility portfolio
min_vol_idx = np.argmin(portfolio_risks)
weights_min_vol = portfolio_weights[min_vol_idx]
ret_min_vol = portfolio_returns[min_vol_idx]
risk_min_vol = portfolio_risks[min_vol_idx]
sharpe_min_vol = portfolio_sharpe[min_vol_idx]

print(f"\n📈 Minimum Volatility Portfolio:")
print(f"  Expected Return: {ret_min_vol:.2%}")
print(f"  Expected Volatility: {risk_min_vol:.2%}")
print(f"  Sharpe Ratio: {sharpe_min_vol:.2f}")
print("  Weights:")
for ticker, weight in zip(tickers, weights_min_vol):
    print(f"    {ticker}: {weight:.2%}")

# ============================================================================
# STEP 6: VISUALIZE EFFICIENT FRONTIER
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 6: VISUALIZE EFFICIENT FRONTIER")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 8))

# Plot all random portfolios (scatter)
scatter = ax.scatter(portfolio_risks, portfolio_returns, 
                     c=portfolio_sharpe, cmap='viridis', 
                     alpha=0.5, s=10)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Sharpe Ratio')

# Plot individual assets
for i, ticker in enumerate(tickers):
    ax.scatter(np.sqrt(cov_matrix.iloc[i, i]), expected_returns_array[i], 
               s=200, label=ticker, zorder=5, edgecolors='black', linewidth=2)

# Plot max Sharpe portfolio
ax.scatter(risk_max_sharpe, ret_max_sharpe, color='red', s=300, 
           marker='*', label=f'Max Sharpe Ratio\n(Sharpe: {sharpe_max_sharpe:.2f})', 
           zorder=10, edgecolors='black', linewidth=2)

# Plot min volatility portfolio
ax.scatter(risk_min_vol, ret_min_vol, color='green', s=300, 
           marker='D', label=f'Min Volatility\n(Sharpe: {sharpe_min_vol:.2f})', 
           zorder=10, edgecolors='black', linewidth=2)

ax.set_xlabel('Expected Volatility (Risk)')
ax.set_ylabel('Expected Return')
ax.set_title('Efficient Frontier - Portfolio Optimization (Monte Carlo)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'efficient_frontier.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Efficient Frontier visualization saved")

# ============================================================================
# STEP 7: FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 7: FINAL RECOMMENDATION")
print("="*60)

# Choose Max Sharpe portfolio as recommended
recommended_weights = weights_max_sharpe
recommended_return = ret_max_sharpe
recommended_risk = risk_max_sharpe
recommended_sharpe = sharpe_max_sharpe

print("""
📋 GMF Investments - Portfolio Recommendation

Based on our analysis using Modern Portfolio Theory (MPT) with:
- TSLA: Forecasted returns from LSTM model (12-month forecast)
- BND: Historical average returns
- SPY: Historical average returns

🎯 RECOMMENDED PORTFOLIO: Maximum Sharpe Ratio Portfolio

This portfolio offers the best risk-adjusted returns for our clients.

📊 Portfolio Allocation:
""")

for ticker, weight in zip(tickers, recommended_weights):
    print(f"  {ticker}: {weight:.2%}")

print(f"""
📈 Portfolio Metrics:
  Expected Annual Return: {recommended_return:.2%}
  Expected Annual Volatility: {recommended_risk:.2%}
  Sharpe Ratio: {recommended_sharpe:.2f}

💡 Justification:
  - This portfolio maximizes return per unit of risk
  - Provides optimal balance between TSLA's growth potential and diversification
  - BND provides stability, SPY provides market exposure
  - TSLA allocation captures upside potential

⚠️ Important Considerations:
  - This is a model-based recommendation
  - Actual performance may differ
  - Consider client's risk tolerance and investment horizon
  - Rebalance periodically
  - Monitor market conditions regularly
""")

# Save recommended portfolio
recommendation_df = pd.DataFrame({
    'Asset': tickers,
    'Recommended_Weight': recommended_weights
})
recommendation_df.to_csv(os.path.join(DATA_DIR, 'recommended_portfolio.csv'), index=False)

# Create portfolio comparison table
portfolio_comparison = pd.DataFrame({
    'Asset': tickers,
    'Max_Sharpe_Weights': weights_max_sharpe,
    'Min_Vol_Weights': weights_min_vol
})
portfolio_comparison.to_csv(os.path.join(DATA_DIR, 'portfolio_weights_comparison.csv'), index=False)

# ============================================================================
# STEP 8: SUMMARY
# ============================================================================

print("\n" + "="*60)
print("📊 TASK 4 COMPLETE - FINAL SUMMARY")
print("="*60)

print("""
✅ Task 4 Completed Successfully!

📊 Key Outputs:
  1. Expected Returns calculated for all assets
  2. Covariance matrix computed
  3. Efficient Frontier generated using Monte Carlo simulation
  4. Maximum Sharpe Ratio portfolio identified
  5. Minimum Volatility portfolio identified
  6. Portfolio recommendation provided

📁 Files Generated:
  - efficient_frontier.png
  - covariance_heatmap.png
  - recommended_portfolio.csv
  - portfolio_weights_comparison.csv

💡 Next Step: Task 5 - Strategy Backtesting
  - Validate portfolio strategy on historical data
  - Compare against benchmark
  - Evaluate performance metrics
""")

print("\n" + "="*60)
print("🎯 TASK 4 COMPLETED SUCCESSFULLY!")
print("="*60)