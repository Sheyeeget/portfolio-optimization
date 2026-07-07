"""
Task 3: Forecast Future Market Trends (FIXED)
GMF Investments - Portfolio Optimization Challenge
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
print("🚀 TASK 3: FORECAST FUTURE MARKET TRENDS")
print("="*60)
print(f"📁 Data directory: {DATA_DIR}")

# ============================================================================
# STEP 1: LOAD DATA AND BEST MODEL
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 1: LOAD DATA AND BEST MODEL")
print("="*60)

# Load cleaned data
data_path = os.path.join(DATA_DIR, 'adj_close_prices_clean.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"✅ Loaded {len(df)} rows of data")

# Focus on TSLA
tsla = df['TSLA'].copy()
print(f"✅ TSLA data shape: {tsla.shape}")
print(f"📅 Date range: {tsla.index.min()} to {tsla.index.max()}")

# Load model comparison to find best model
comparison_path = os.path.join(DATA_DIR, 'model_comparison.csv')
comparison_df = pd.read_csv(comparison_path, index_col=0)
print("\n📊 Model Comparison:")
print(comparison_df)

best_model = comparison_df['MAE'].idxmin()
print(f"\n🏆 Best Model: {best_model} (MAE: ${comparison_df.loc[best_model, 'MAE']:.2f})")

# ============================================================================
# STEP 2: GENERATE FUTURE FORECASTS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 2: GENERATE FUTURE FORECASTS")
print("="*60)

forecast_days = 252  # 12 months (~252 trading days)
print(f"📈 Forecast horizon: {forecast_days} trading days (~12 months)")

# ============================================================================
# STEP 3: LSTM MODEL FOR FUTURE FORECASTING
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 3: LSTM FUTURE FORECAST")
print("="*60)

print("📈 Using LSTM model for future forecasts...")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Prepare data
scaler = MinMaxScaler()
tsla_scaled = scaler.fit_transform(tsla.values.reshape(-1, 1))

seq_length = 60  # Use 60 days to predict next day

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(tsla_scaled, seq_length)
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"Training LSTM on {len(X)} sequences...")

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = lstm_model.fit(
    X, y, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.1, 
    callbacks=[early_stop],
    verbose=1
)
print("✅ LSTM model trained successfully!")

# Generate future predictions iteratively
last_sequence = tsla_scaled[-seq_length:].copy()
future_predictions = []

print(f"🔄 Generating {forecast_days} future predictions...")
for i in range(forecast_days):
    pred_scaled = lstm_model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
    future_predictions.append(pred_scaled[0, 0])
    last_sequence = np.concatenate([last_sequence[1:], pred_scaled])

# Inverse transform predictions to original price scale
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Create date index for future dates
last_date = tsla.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')

# Create forecast Series
future_forecast = pd.Series(future_prices, index=future_dates)

print(f"✅ Generated {len(future_forecast)} future forecasts")
print(f"📅 Date range: {future_forecast.index.min()} to {future_forecast.index.max()}")

# ============================================================================
# STEP 4: CALCULATE CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 4: CALCULATE CONFIDENCE INTERVALS")
print("="*60)

# Load LSTM forecast from Task 2 to calculate error
lstm_forecast_path = os.path.join(DATA_DIR, 'lstm_forecast.csv')
lstm_forecast = pd.read_csv(lstm_forecast_path, index_col=0, parse_dates=True)

# Align actual and forecasted values
lstm_actual = tsla.loc[lstm_forecast.index]
min_len = min(len(lstm_actual), len(lstm_forecast))
lstm_actual = lstm_actual[:min_len]
lstm_forecast_aligned = lstm_forecast[:min_len]

# Calculate percentage errors
lstm_error = (lstm_actual - lstm_forecast_aligned) / lstm_actual

# FIX: Extract scalar value from Series
std_error = lstm_error.std()
if hasattr(std_error, 'iloc'):
    std_error = std_error.iloc[0]  # Extract scalar if it's a Series
elif hasattr(std_error, 'values'):
    std_error = std_error.values[0]  # Alternative extraction

confidence_interval = 1.96 * std_error  # 95% confidence

print(f"LSTM Error Std Dev: {std_error:.4f}")
print(f"95% Confidence Interval: ±{confidence_interval:.2%}")

# Create confidence bands that expand over time
ci_multiplier = np.sqrt(np.arange(1, len(future_forecast) + 1) / 252)
ci_values = confidence_interval * ci_multiplier

# Ensure arrays have same length
if len(ci_values) != len(future_forecast):
    ci_values = ci_values[:len(future_forecast)]

# Calculate upper and lower bands
upper_band = future_forecast * (1 + ci_values)
lower_band = future_forecast * (1 - ci_values)

print("✅ Confidence intervals calculated successfully")

# ============================================================================
# STEP 5: VISUALIZE FORECASTS WITH CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 5: VISUALIZE FORECASTS")
print("="*60)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Recent history with future forecast
ax = axes[0]
lookback = 504  # ~2 years of historical data

# Historical data (last 2 years)
ax.plot(tsla.index[-lookback:], tsla.values[-lookback:], 
        label='Historical Price', linewidth=2, color='blue')

# Future forecast
ax.plot(future_forecast.index, future_forecast, 
        label='LSTM Forecast', linewidth=2, color='green')

# Confidence interval bands
ax.fill_between(future_forecast.index, lower_band, upper_band, 
                alpha=0.3, color='green', label='95% Confidence Interval')

# Vertical line at forecast start
ax.axvline(x=tsla.index[-1], color='red', linestyle='--', 
           alpha=0.7, linewidth=2, label='Forecast Start')

ax.set_title('TSLA Stock Price - 12-Month Forecast with Confidence Intervals', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Zoom on forecast period
ax = axes[1]
ax.plot(future_forecast.index, future_forecast, 
        label='LSTM Forecast', linewidth=2, color='green')
ax.fill_between(future_forecast.index, lower_band, upper_band, 
                alpha=0.3, color='green', label='95% Confidence Interval')
ax.set_title('TSLA - 12-Month Forecast (Zoom)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
forecast_path = os.path.join(DATA_DIR, 'future_forecast_12months.png')
plt.savefig(forecast_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Forecast visualization saved to: {forecast_path}")

# ============================================================================
# STEP 6: TREND ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 6: TREND ANALYSIS")
print("="*60)

# Calculate key metrics
start_price = future_forecast.iloc[0]
end_price = future_forecast.iloc[-1]
price_change = end_price - start_price
price_change_pct = (price_change / start_price) * 100

max_price = future_forecast.max()
min_price = future_forecast.min()
avg_price = future_forecast.mean()

# Determine trend
if price_change_pct > 5:
    trend = "📈 UPWARD"
elif price_change_pct < -5:
    trend = "📉 DOWNWARD"
else:
    trend = "➡️ STABLE"

print(f"\n📈 Forecast Analysis:")
print(f"  Starting Price: ${start_price:.2f}")
print(f"  Ending Price:   ${end_price:.2f}")
print(f"  Total Change:   ${price_change:.2f} ({price_change_pct:.2f}%)")
print(f"  Highest Price:  ${max_price:.2f}")
print(f"  Lowest Price:   ${min_price:.2f}")
print(f"  Average Price:  ${avg_price:.2f}")
print(f"\n📊 Trend Direction: {trend}")

# Confidence interval analysis
ci_width_start = upper_band.iloc[0] - lower_band.iloc[0]
ci_width_end = upper_band.iloc[-1] - lower_band.iloc[-1]
ci_expansion = ci_width_end / ci_width_start if ci_width_start > 0 else 1

print(f"\n📊 Confidence Interval Analysis:")
print(f"  Initial CI Width:  ${ci_width_start:.2f}")
print(f"  Final CI Width:    ${ci_width_end:.2f}")
print(f"  CI Expansion:      {ci_expansion:.2f}x")

if ci_expansion > 2:
    print(f"  ⚠️ Confidence intervals widen significantly over time")
    print(f"     → Short-term forecasts are more reliable than long-term")
else:
    print(f"  ✅ Confidence intervals remain relatively stable")

# ============================================================================
# STEP 7: OPPORTUNITIES AND RISKS ASSESSMENT
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 7: OPPORTUNITIES AND RISKS ASSESSMENT")
print("="*60)

# Identify opportunities based on trend
if price_change_pct > 5:
    opportunities = [
        f"📈 Expected price appreciation of {price_change_pct:.2f}% over the next 12 months",
        "💰 Potential for capital gains if trend continues",
        "📊 Positive momentum may attract more investors",
        "🏆 Strong performance relative to market expectations"
    ]
elif price_change_pct < -5:
    opportunities = [
        "📉 Lower prices may present buying opportunities",
        "📊 Potential for mean reversion if oversold",
        "💰 Accumulation opportunity for long-term investors",
        "🔍 Value hunting opportunity in a down market"
    ]
else:
    opportunities = [
        "📊 Stable prices provide predictable environment",
        "💰 Consistent returns with lower volatility",
        "📈 Opportunity for steady portfolio growth",
        "⚖️ Balanced risk-reward profile"
    ]

# Identify risks (same for all scenarios)
risks = [
    "⚠️ High uncertainty (wide confidence intervals)",
    "📊 Market volatility could impact actual performance",
    "⚠️ External factors (economic, regulatory) not captured",
    "📉 Actual prices may deviate significantly from forecast",
    "🌍 Geopolitical events could disrupt market conditions",
    "🏭 Company-specific risks (competition, innovation, leadership)"
]

print("\n📈 Market Opportunities:")
for opp in opportunities:
    print(f"  {opp}")

print("\n⚠️ Market Risks:")
for risk in risks:
    print(f"  {risk}")

# ============================================================================
# STEP 8: SAVE FORECASTS FOR TASK 4
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 8: SAVE FORECASTS FOR TASK 4")
print("="*60)

# Save future forecast
future_forecast_path = os.path.join(DATA_DIR, 'future_forecast_12months.csv')
future_forecast.to_csv(future_forecast_path)
print(f"✅ Future forecast saved to: {future_forecast_path}")

# Save confidence intervals
ci_df = pd.DataFrame({
    'Forecast': future_forecast,
    'Upper_Band': upper_band,
    'Lower_Band': lower_band
})
ci_path = os.path.join(DATA_DIR, 'confidence_intervals.csv')
ci_df.to_csv(ci_path)
print(f"✅ Confidence intervals saved to: {ci_path}")

# Save LSTM training history
history_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Training_Loss': history.history['loss'],
    'Validation_Loss': history.history['val_loss']
})
history_df.to_csv(os.path.join(DATA_DIR, 'lstm_training_history_future.csv'), index=False)

# ============================================================================
# STEP 9: FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("📊 TASK 3 COMPLETE - FINAL SUMMARY")
print("="*60)

print(f"""
Model Used: {best_model}
Forecast Horizon: 12 months ({forecast_days} trading days)

Key Findings:
-------------
1. Price Trend: {trend}
2. Expected Change: {price_change_pct:.2f}%
3. Price Range: ${min_price:.2f} - ${max_price:.2f}
4. Confidence Interval Expansion: {ci_expansion:.2f}x

Opportunities:
-------------
{chr(10).join(opportunities)}

Risks:
------
{chr(10).join(risks)}

Recommendations:
---------------
✅ Use this forecast as ONE input, not the sole decision-maker
✅ Consider multiple scenarios (bull, bear, base)
✅ Update forecasts regularly with new data
✅ Combine with fundamental analysis for better decisions
✅ Short-term forecasts (1-3 months) are more reliable than long-term
✅ Confidence intervals widen → longer forecasts less certain
""")

print("\n" + "="*60)
print("🎯 TASK 3 COMPLETED SUCCESSFULLY!")
print("="*60)

print("\n📋 Files Generated:")
print("  📄 future_forecast_12months.png")
print("  📄 future_forecast_12months.csv")
print("  📄 confidence_intervals.csv")
print("  📄 lstm_training_history_future.csv")

print("\n📂 All files saved in:", DATA_DIR)

print("\n💡 Next Step: Task 4 - Portfolio Optimization using Modern Portfolio Theory (MPT)")