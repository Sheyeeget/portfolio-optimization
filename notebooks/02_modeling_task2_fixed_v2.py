"""
Task 2: Build Time Series Forecasting Models (FIXED V2)
GMF Investments - Portfolio Optimization Challenge
Goal: Build ARIMA/SARIMA and LSTM models to predict TSLA stock prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
from datetime import datetime

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
print("🚀 TASK 2: TIME SERIES FORECASTING MODELS")
print("="*60)
print(f"📁 Data directory: {DATA_DIR}")

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 1: LOAD AND PREPARE DATA")
print("="*60)

# Load cleaned data
data_path = os.path.join(DATA_DIR, 'adj_close_prices_clean.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"✅ Loaded {len(df)} rows of data")

# Focus on TSLA (we'll predict TSLA)
tsla = df['TSLA'].copy()
print(f"✅ TSLA data shape: {tsla.shape}")
print(f"📅 Date range: {tsla.index.min()} to {tsla.index.max()}")

# Split data chronologically (NO SHUFFLING!)
train_end = '2024-12-31'
test_start = '2025-01-01'

train = tsla.loc[:train_end].copy()
test = tsla.loc[test_start:].copy()

print(f"\n📊 Data Split:")
print(f"  Training: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"  Testing:  {test.index.min()} to {test.index.max()} ({len(test)} days)")

# Plot the split
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train.index, train, label='Training Data', linewidth=2, color='blue')
ax.plot(test.index, test, label='Test Data', linewidth=2, color='green')
split_date = pd.to_datetime('2024-12-31')
ax.axvline(x=split_date, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Split Point')
ax.set_title('TSLA Stock Price - Train/Test Split', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'train_test_split.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Split visualization saved")

# ============================================================================
# STEP 2: ARIMA/SARIMA MODEL
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 2: ARIMA/SARIMA MODEL")
print("="*60)

# Import ARIMA libraries
try:
    import pmdarima as pm
    from pmdarima import auto_arima
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    print("✅ All ARIMA libraries imported successfully")
except ImportError as e:
    print(f"⚠️ Error importing libraries: {e}")
    print("Installing pmdarima...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pmdarima'])
    import pmdarima as pm
    from pmdarima import auto_arima
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

# Calculate returns for ARIMA (stationary data)
print("\n📈 Calculating returns for ARIMA modeling...")
train_returns = train.pct_change().dropna()
test_returns = test.pct_change().dropna()

print(f"Train returns: {len(train_returns)} days")
print(f"Test returns: {len(test_returns)} days")

# Test stationarity of returns
print("\n🔍 Testing stationarity of returns...")
result = adfuller(train_returns.dropna())
print(f"  ADF Statistic: {result[0]:.6f}")
print(f"  p-value: {result[1]:.6f}")
print(f"  Stationary: {result[1] < 0.05}")

# Find optimal ARIMA parameters using auto_arima
print("\n🔍 Finding optimal ARIMA parameters (this may take 1-2 minutes)...")
start_time = time.time()

try:
    auto_model = auto_arima(
        train_returns,
        start_p=0, max_p=5,
        start_d=0, max_d=2,
        start_q=0, max_q=5,
        seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic',
        n_fits=50
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Auto-ARIMA completed in {elapsed:.1f} seconds")
    print(f"✅ Optimal ARIMA order: {auto_model.order}")
    print(f"✅ Optimal SARIMA order: {auto_model.seasonal_order}")
    
    # Fit the model
    p, d, q = auto_model.order
    print(f"\n📈 Fitting ARIMA({p},{d},{q}) model...")
    arima_model = ARIMA(train_returns, order=(p, d, q))
    arima_fit = arima_model.fit()
    print("✅ ARIMA model fitted successfully")
    print(arima_fit.summary())
    
except Exception as e:
    print(f"⚠️ Auto-ARIMA failed: {e}")
    print("Using default ARIMA(1,0,1) model...")
    
    # Fallback to a simple ARIMA model
    arima_model = ARIMA(train_returns, order=(1, 0, 1))
    arima_fit = arima_model.fit()
    print("✅ ARIMA(1,0,1) model fitted as fallback")

# Generate return forecasts
print("\n📈 Generating ARIMA return forecasts...")
forecast_steps = len(test)  # Use len(test) not len(test_returns)
forecast_returns = arima_fit.forecast(steps=forecast_steps)

# Convert return forecasts to price forecasts
# Starting from last training price
last_train_price = train.iloc[-1]
forecast_prices = [last_train_price]

for ret in forecast_returns:
    forecast_prices.append(forecast_prices[-1] * (1 + ret))

# Remove the first element (the starting price)
forecast_prices = np.array(forecast_prices[1:])

# Ensure we have exactly len(test) values
if len(forecast_prices) > len(test):
    forecast_prices = forecast_prices[:len(test)]
elif len(forecast_prices) < len(test):
    # Pad with NaN if needed (shouldn't happen with correct steps)
    padding = [np.nan] * (len(test) - len(forecast_prices))
    forecast_prices = np.concatenate([forecast_prices, padding])

# Create Series for plotting
arima_forecast = pd.Series(forecast_prices, index=test.index)

print(f"✅ Generated {len(arima_forecast)} ARIMA forecasted days")

# ============================================================================
# STEP 3: LSTM MODEL
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 3: LSTM MODEL")
print("="*60)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    print(f"✅ TensorFlow version: {tf.__version__}")
    print("✅ All LSTM libraries imported successfully")
except ImportError as e:
    print(f"⚠️ Error importing TensorFlow: {e}")
    print("Installing TensorFlow...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tensorflow'])
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

# Prepare data for LSTM
print("\n🔧 Preparing data for LSTM...")

# Scale the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60  # Use 60 days to predict the next day

X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Reshape for LSTM input (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Build LSTM model
print("\n🏗️ Building LSTM model...")
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.summary()

# Train LSTM
print("\n🚀 Training LSTM model...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print("✅ LSTM model trained successfully!")

# Generate LSTM predictions
print("\n📈 Generating LSTM predictions...")
lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# Align with test dates
lstm_dates = test.index[seq_length:]
lstm_forecast = pd.Series(lstm_pred.flatten(), index=lstm_dates)

print(f"✅ Generated {len(lstm_forecast)} LSTM predictions")

# ============================================================================
# STEP 4: MODEL EVALUATION
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 4: MODEL EVALUATION")
print("="*60)

from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(actual, predicted, model_name):
    """Calculate and print evaluation metrics."""
    # Align indices
    common_idx = actual.index.intersection(predicted.index)
    actual_aligned = actual[common_idx]
    predicted_aligned = predicted[common_idx]
    
    # Remove any NaN values
    mask = ~(np.isnan(actual_aligned) | np.isnan(predicted_aligned))
    actual_aligned = actual_aligned[mask]
    predicted_aligned = predicted_aligned[mask]
    
    if len(actual_aligned) == 0:
        print(f"\n{model_name}: No overlapping data for evaluation")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
    
    mae = mean_absolute_error(actual_aligned, predicted_aligned)
    rmse = np.sqrt(mean_squared_error(actual_aligned, predicted_aligned))
    mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
    
    print(f"\n{model_name} Metrics:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  N:    {len(actual_aligned)} days")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# ARIMA evaluation
arima_metrics = calculate_metrics(test, arima_forecast, "ARIMA")

# LSTM evaluation
lstm_metrics = calculate_metrics(test, lstm_forecast, "LSTM")

# Comparison table
comparison_df = pd.DataFrame({
    'ARIMA': arima_metrics,
    'LSTM': lstm_metrics
}).T.round(2)

print("\n📊 Model Comparison:")
print("="*60)
print(comparison_df)

# Save comparison
comparison_path = os.path.join(DATA_DIR, 'model_comparison.csv')
comparison_df.to_csv(comparison_path)
print(f"\n💾 Model comparison saved to: {comparison_path}")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 5: VISUALIZATIONS")
print("="*60)

# Plot 1: Full forecast comparison
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

ax = axes[0]
ax.plot(train.index, train, label='Training Data', linewidth=1.5, color='blue')
ax.plot(test.index, test, label='Actual Test Data', linewidth=2, color='green')
ax.plot(arima_forecast.index, arima_forecast, label='ARIMA Forecast', linewidth=1.5, color='orange', linestyle='--')
ax.plot(lstm_forecast.index, lstm_forecast, label='LSTM Forecast', linewidth=1.5, color='red', linestyle='--')
split_date = pd.to_datetime('2024-12-31')
ax.axvline(x=split_date, color='black', linestyle='--', alpha=0.5, linewidth=2)
ax.set_title('TSLA Stock Price - Model Forecast Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Zoom on test period
ax = axes[1]
ax.plot(test.index, test, label='Actual', linewidth=2, color='green')
ax.plot(arima_forecast.index, arima_forecast, label='ARIMA', linewidth=1.5, color='orange', linestyle='--')
ax.plot(lstm_forecast.index, lstm_forecast, label='LSTM', linewidth=1.5, color='red', linestyle='--')
ax.set_title('TSLA Stock Price - Test Period Forecasts (2025-2026)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
forecast_comp_path = os.path.join(DATA_DIR, 'model_forecasts_comparison.png')
plt.savefig(forecast_comp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Forecast comparison saved to: {forecast_comp_path}")

# Plot LSTM training history
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_title('LSTM Training History', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
lstm_history_path = os.path.join(DATA_DIR, 'lstm_training_history.png')
plt.savefig(lstm_history_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ LSTM training history saved to: {lstm_history_path}")

# ============================================================================
# STEP 6: SAVE FORECASTS FOR TASK 3
# ============================================================================

print("\n" + "="*60)
print("📊 STEP 6: SAVE FORECASTS FOR TASK 3")
print("="*60)

# Save ARIMA forecast
arima_forecast_path = os.path.join(DATA_DIR, 'arima_forecast.csv')
arima_forecast.to_csv(arima_forecast_path)
print(f"✅ ARIMA forecast saved to: {arima_forecast_path}")

# Save LSTM forecast
lstm_forecast_path = os.path.join(DATA_DIR, 'lstm_forecast.csv')
lstm_forecast.to_csv(lstm_forecast_path)
print(f"✅ LSTM forecast saved to: {lstm_forecast_path}")

# ============================================================================
# STEP 7: SUMMARY
# ============================================================================

print("\n" + "="*60)
print("📊 TASK 2 SUMMARY")
print("="*60)

print("\nModel Performance Comparison:")
print("="*60)
print(comparison_df)

# Determine which model performed better
if not comparison_df['MAE'].isna().all():
    best_model = comparison_df['MAE'].idxmin()
    best_mae = comparison_df.loc[best_model, 'MAE']
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Best MAE: ${best_mae:.2f}")
else:
    print("\n⚠️ Could not determine best model (no valid metrics)")

print("\n" + "="*60)
print("🎯 TASK 2 COMPLETED SUCCESSFULLY!")
print("="*60)

print("\n📋 Files Generated:")
print("  - train_test_split.png")
print("  - model_forecasts_comparison.png")
print("  - lstm_training_history.png")
print("  - model_comparison.csv")
print("  - arima_forecast.csv")
print("  - lstm_forecast.csv")

print("\n📂 All files saved in:", DATA_DIR)