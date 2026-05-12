# ============================================================
# Forecast People_Next10 using LSTM (TensorFlow)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# 1. Load Excel Data
# ============================================================
file_name = r'D:\Senior Project\Data\Data_Train_3_Year.xlsx'
sheets = ['Year_2023', 'Year_2024', 'Year_2025']

data_frames = []

for sheet in sheets:
    try:
        df = pd.read_excel(file_name, sheet_name=sheet)
        data_frames.append(df)
        print(f"โหลดข้อมูล {sheet} สำเร็จ: {len(df)} แถว")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลด {sheet}: {e}")

Data = pd.concat(data_frames, ignore_index=True)
Data = Data.dropna().reset_index(drop=True)

# ============================================================
# 2. Feature Engineering
# ============================================================
def encode_cyclical(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

Data = encode_cyclical(Data, 'Mn', 60)
Data = encode_cyclical(Data, 'Hr', 24)
Data = encode_cyclical(Data, 'DoW', 7)
Data = encode_cyclical(Data, 'DoM', 31)
Data = encode_cyclical(Data, 'Month', 12)

Data['IsHoliday'] = Data['IsHoliday'].astype(float)
Data['Schedule'] = Data['Schedule'].astype(float)
Data['People'] = Data['People'].astype(float)



# Rolling
Data['People_RollMean_3'] = Data['People'].rolling(3).mean()
Data['People_RollMean_6'] = Data['People'].rolling(6).mean()

Data = Data.dropna().reset_index(drop=True)

# ============================================================
# Label
# ============================================================
Data['People_Next10'] = Data['People'].shift(-10)
Data = Data.dropna().reset_index(drop=True)

# ============================================================
# Features
# ============================================================
features = [
    'DoW_sin','DoW_cos',
    'Hr_sin','Hr_cos',
    'Mn_sin','Mn_cos',
    'IsHoliday',
    'Schedule',
    'People'
]

target = 'People_Next10'

# ============================================================
# Scaling
# ============================================================
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

Data[features] = scaler_x.fit_transform(Data[features])
Data[[target]] = scaler_y.fit_transform(Data[[target]])

# ============================================================
# Sequence Generator
# ============================================================
def create_sequences(data, feature_cols, target_col, seq_length):
    xs, ys = [], []
    data_feat = data[feature_cols].values
    data_targ = data[target_col].values

    for i in range(len(data) - seq_length):
        x = data_feat[i:i + seq_length]
        y = data_targ[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

SEQ_LENGTH = 6

X, y = create_sequences(Data, features, target, SEQ_LENGTH)

# ============================================================
# Train/Test Split
# ============================================================
train_size = int(len(X) * 0.7)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ============================================================
# 8. LSTM Model (TensorFlow)
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ============================================================
# 8. LSTM Model (Sequential Version)
# ============================================================
model = Sequential([
    LSTM(32,return_sequences=False ,input_shape=(SEQ_LENGTH, len(features))),
    Dropout(0.2),
    Dense(1)
])
# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

model.summary()

# ============================================================
# Callbacks (แทน scheduler + early stopping)
# ============================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=5
)

# ============================================================
# 11. Training
# ============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=300,
    batch_size=32,
    shuffle=False,
    #callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid(alpha=0.3)

# Prediction
pred = model.predict(X_test)

pred_real = scaler_y.inverse_transform(pred)
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1,1))

plt.subplot(1,2,2)
plt.plot(y_test_real[:150], label='Actual')
plt.plot(pred_real[:150], '--', label='Predicted')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Metrics
# ============================================================
mae = mean_absolute_error(y_test_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
r2 = r2_score(y_test_real, pred_real)

mape = np.mean(
    np.abs((y_test_real - pred_real) / np.clip(y_test_real, 1e-8, None))
) * 100

accuracy = 100 - mape

print("\n================ Final Model Performance ================")
print(f"MAE   : {mae:.4f} คน")
print(f"RMSE  : {rmse:.4f} คน")
print(f"R²    : {r2:.4f}")
print(f"MAPE  : {mape:.2f}%")
print(f"Accuracy Approximation : {accuracy:.2f}%")
print("========================================================")

# ============================================================
# Save Model
# ============================================================
'''
model.save("best_AI_TES_Model_tf.h5")
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
'''
