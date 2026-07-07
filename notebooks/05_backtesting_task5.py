"""
Task 5: Strategy Backtesting
GMF Investments - Portfolio Optimization Challenge
Goal: Validate portfolio strategy against benchmark
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
print("🚀 TASK 5: STRATEGY BACKTESTING")
print("="*60)
print(f"📁 Data directory: {DATA_DIR}")

# ============================================================================
# STEP 1: LOAD DATA AND PORTFOLIO WEIGHTS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 1: LOAD DATA AND PORTFOLIO WEIGHTS")
print("="*60)

# Load cleaned data
data_path = os.path.join(DATA_DIR, 'adj_close_prices_clean.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"✅ Loaded {len(df)} rows of data")

# Calculate returns
returns = df.pct_change().dropna()
print(f"✅ Returns shape: {returns.shape}")

# Load recommended portfolio weights from Task 4
weights_path = os.path.join(DATA_DIR, 'recommended_portfolio.csv')
recommended_weights = pd.read_csv(weights_path)
print("\n📊 Recommended Portfolio Weights:")
print(recommended_weights)

# Extract weights as dictionary
weights_dict = dict(zip(recommended_weights['Asset'], recommended_weights['Recommended_Weight']))
print("\n📊 Weights Dictionary:")
print(weights_dict)

# ============================================================================
# STEP 2: DEFINE BACKTESTING PERIOD
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 2: DEFINE BACKTESTING PERIOD")
print("="*60)

# Use the last year of data for backtesting
backtest_start = '2025-01-01'
backtest_end = '2026-06-30'

# Filter returns for backtesting period
backtest_returns = returns.loc[backtest_start:backtest_end]
backtest_prices = df.loc[backtest_start:backtest_end]

print(f"📅 Backtesting Period: {backtest_start} to {backtest_end}")
print(f"📊 Backtesting Days: {len(backtest_returns)}")

# ============================================================================
# STEP 3: DEFINE BENCHMARK
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 3: DEFINE BENCHMARK")
print("="*60)

# Benchmark: 60% SPY / 40% BND (common balanced portfolio)
benchmark_weights = {
    'SPY': 0.60,
    'BND': 0.40,
    'TSLA': 0.00
}

print("\n📊 Benchmark Weights (60/40):")
for asset, weight in benchmark_weights.items():
    print(f"  {asset}: {weight:.2%}")

# ============================================================================
# STEP 4: SIMULATE STRATEGY PERFORMANCE
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 4: SIMULATE STRATEGY PERFORMANCE")
print("="*60)

# Define function to calculate portfolio returns
def calculate_portfolio_returns(returns, weights):
    """Calculate weighted portfolio returns."""
    # Ensure we have all assets in weights
    portfolio_returns = pd.Series(0, index=returns.index)
    
    for asset, weight in weights.items():
        if asset in returns.columns:
            portfolio_returns += returns[asset] * weight
    
    return portfolio_returns

# Calculate strategy returns
strategy_returns = calculate_portfolio_returns(backtest_returns, weights_dict)

# Calculate benchmark returns
benchmark_returns = calculate_portfolio_returns(backtest_returns, benchmark_weights)

print(f"✅ Strategy Returns: {len(strategy_returns)} days")
print(f"✅ Benchmark Returns: {len(benchmark_returns)} days")

# ============================================================================
# STEP 5: PERFORMANCE METRICS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 5: PERFORMANCE METRICS")
print("="*60)

def calculate_metrics(returns, name, risk_free_rate=0.02):
    """Calculate comprehensive performance metrics."""
    # Remove any NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        return None
    
    # Calculate total return
    total_return = (1 + returns).prod() - 1
    
    # Calculate annualized return
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    # Calculate annualized volatility
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    excess_return = returns.mean() - risk_free_rate / 252
    sharpe_ratio = (excess_return * np.sqrt(252)) / returns.std() if returns.std() > 0 else 0
    
    # Calculate maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (returns > 0).mean()
    
    # Calculate positive/negative days
    positive_days = (returns > 0).sum()
    negative_days = (returns < 0).sum()
    
    return {
        'name': name,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'positive_days': positive_days,
        'negative_days': negative_days,
        'n_days': len(returns)
    }

# Calculate metrics for both portfolios
strategy_metrics = calculate_metrics(strategy_returns, 'Strategy')
benchmark_metrics = calculate_metrics(benchmark_returns, 'Benchmark')

# Create comparison DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['Total Return', 'Annualized Return', 'Annualized Volatility', 
               'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Positive Days', 'Negative Days'],
    'Strategy': [
        f"{strategy_metrics['total_return']:.2%}",
        f"{strategy_metrics['annualized_return']:.2%}",
        f"{strategy_metrics['annualized_volatility']:.2%}",
        f"{strategy_metrics['sharpe_ratio']:.2f}",
        f"{strategy_metrics['max_drawdown']:.2%}",
        f"{strategy_metrics['win_rate']:.2%}",
        f"{strategy_metrics['positive_days']}",
        f"{strategy_metrics['negative_days']}"
    ],
    'Benchmark': [
        f"{benchmark_metrics['total_return']:.2%}",
        f"{benchmark_metrics['annualized_return']:.2%}",
        f"{benchmark_metrics['annualized_volatility']:.2%}",
        f"{benchmark_metrics['sharpe_ratio']:.2f}",
        f"{benchmark_metrics['max_drawdown']:.2%}",
        f"{benchmark_metrics['win_rate']:.2%}",
        f"{benchmark_metrics['positive_days']}",
        f"{benchmark_metrics['negative_days']}"
    ]
})

print("\n📊 Performance Comparison:")
print("="*80)
print(metrics_df.to_string(index=False))

# ============================================================================
# STEP 6: VISUALIZE PERFORMANCE
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 6: VISUALIZE PERFORMANCE")
print("="*60)

# Calculate cumulative returns
strategy_cumulative = (1 + strategy_returns).cumprod()
benchmark_cumulative = (1 + benchmark_returns).cumprod()

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Cumulative Returns Comparison
ax = axes[0]
ax.plot(strategy_cumulative.index, strategy_cumulative, 
        label=f'Strategy Portfolio (Sharpe: {strategy_metrics["sharpe_ratio"]:.2f})', 
        linewidth=2, color='blue')
ax.plot(benchmark_cumulative.index, benchmark_cumulative, 
        label=f'Benchmark (60/40) (Sharpe: {benchmark_metrics["sharpe_ratio"]:.2f})', 
        linewidth=2, color='green')
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax.set_title('Cumulative Returns - Strategy vs Benchmark', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Drawdown Comparison
ax = axes[1]
# Calculate drawdowns
strategy_drawdown = (strategy_cumulative / strategy_cumulative.expanding().max()) - 1
benchmark_drawdown = (benchmark_cumulative / benchmark_cumulative.expanding().max()) - 1

ax.fill_between(strategy_drawdown.index, 0, strategy_drawdown, 
                label='Strategy Drawdown', alpha=0.5, color='blue')
ax.fill_between(benchmark_drawdown.index, 0, benchmark_drawdown, 
                label='Benchmark Drawdown', alpha=0.5, color='green')
ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'backtest_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Backtest comparison visualization saved")

# ============================================================================
# STEP 7: SAVE METRICS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 7: SAVE METRICS")
print("="*60)

# Save metrics to CSV
metrics_export = pd.DataFrame({
    'Metric': ['Total Return', 'Annualized Return', 'Annualized Volatility', 
               'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Positive Days', 'Negative Days', 'N Days'],
    'Strategy': [
        strategy_metrics['total_return'],
        strategy_metrics['annualized_return'],
        strategy_metrics['annualized_volatility'],
        strategy_metrics['sharpe_ratio'],
        strategy_metrics['max_drawdown'],
        strategy_metrics['win_rate'],
        strategy_metrics['positive_days'],
        strategy_metrics['negative_days'],
        strategy_metrics['n_days']
    ],
    'Benchmark': [
        benchmark_metrics['total_return'],
        benchmark_metrics['annualized_return'],
        benchmark_metrics['annualized_volatility'],
        benchmark_metrics['sharpe_ratio'],
        benchmark_metrics['max_drawdown'],
        benchmark_metrics['win_rate'],
        benchmark_metrics['positive_days'],
        benchmark_metrics['negative_days'],
        benchmark_metrics['n_days']
    ]
})

metrics_export.to_csv(os.path.join(DATA_DIR, 'backtest_metrics.csv'), index=False)
print("✅ Backtest metrics saved to: backtest_metrics.csv")

# ============================================================================
# STEP 8: CONCLUSION
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 8: CONCLUSION")
print("="*60)

# Determine if strategy outperformed benchmark
strategy_better = strategy_metrics['sharpe_ratio'] > benchmark_metrics['sharpe_ratio']

print(f"""
📋 Backtest Results Analysis

Period: {backtest_start} to {backtest_end}

Strategy vs Benchmark:
-----------------------
1. Total Return:     Strategy ({strategy_metrics['total_return']:.2%}) vs 
                     Benchmark ({benchmark_metrics['total_return']:.2%})

2. Annualized Return: Strategy ({strategy_metrics['annualized_return']:.2%}) vs 
                      Benchmark ({benchmark_metrics['annualized_return']:.2%})

3. Sharpe Ratio:     Strategy ({strategy_metrics['sharpe_ratio']:.2f}) vs 
                     Benchmark ({benchmark_metrics['sharpe_ratio']:.2f})

4. Max Drawdown:     Strategy ({strategy_metrics['max_drawdown']:.2%}) vs 
                     Benchmark ({benchmark_metrics['max_drawdown']:.2%})

5. Win Rate:         Strategy ({strategy_metrics['win_rate']:.2%}) vs 
                     Benchmark ({benchmark_metrics['win_rate']:.2%})

📈 Did the strategy outperform the benchmark?
   {'✅ YES - The strategy outperformed the benchmark!' if strategy_better else '❌ NO - The benchmark outperformed the strategy.'}

💡 Key Insights:
   - The strategy {'outperformed' if strategy_better else 'underperformed'} the benchmark in terms of risk-adjusted returns
   - Consider the following factors:
     * TSLA's high volatility contributed to risk
     * Portfolio diversification helped manage drawdowns
     * The model-based approach captured market trends

⚠️ Limitations of this Backtest:
   1. Limited historical period (1.5 years)
   2. Assumes perfect execution (no transaction costs)
   3. Model-based forecasts may not repeat
   4. Does not account for market regime changes

📝 Recommendations:
   1. Consider longer backtesting periods
   2. Include transaction costs in future analysis
   3. Use rolling window forecasts for better adaptation
   4. Combine with fundamental analysis for validation
   5. Regular rebalancing to maintain target weights
""")

print("\n" + "="*60)
print("🎯 TASK 5 COMPLETED SUCCESSFULLY!")
print("="*60)

print("\n📋 Files Generated:")
print("  - backtest_comparison.png")
print("  - backtest_metrics.csv")

print("\n📂 All files saved in:", DATA_DIR)
print("\n🎉 ALL 5 TASKS COMPLETED!")
print("="*60)