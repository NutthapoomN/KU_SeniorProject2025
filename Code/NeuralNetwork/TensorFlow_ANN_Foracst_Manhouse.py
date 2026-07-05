# Forecast Man-Hour Next Day using Neural Network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = r"D:\Senior Project\Data\Data_Train_3_Year.xlsx"
SHEET_NAME = "Train_manhouse"

FEATURE = [
    "DoW",
    "DoM",
    "Month",
    "Schedule",
    "IsHoliday",
    "Term",
    "ManHour_Lag7"
]

TARGET = "Man-House"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
df = df.dropna().reset_index(drop=True)

print("Rows :", len(df))

# ============================================================
# SPLIT BEFORE SCALING
# ============================================================

split_train = int(len(df)*0.70)
split_val = int(len(df)*0.85)

train_df = df.iloc[:split_train]
val_df = df.iloc[split_train:split_val]
test_df = df.iloc[split_val:]

# ============================================================
# SCALE
# ============================================================

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(train_df[FEATURE])
scaler_y.fit(train_df[[TARGET]])

X_train = scaler_x.transform(train_df[FEATURE])
X_val = scaler_x.transform(val_df[FEATURE])
X_test = scaler_x.transform(test_df[FEATURE])

y_train = scaler_y.transform(train_df[[TARGET]])
y_val = scaler_y.transform(val_df[[TARGET]])
y_test = scaler_y.transform(test_df[[TARGET]])

# ============================================================
# MODEL
# ============================================================

model = Sequential([
    Input(shape=(len(FEATURE),)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="mse"
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5
)

# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.grid()
plt.legend()
plt.show()

# ============================================================
# TEST
# ============================================================

pred_scaled = model.predict(X_test, verbose=0)

pred_real = scaler_y.inverse_transform(pred_scaled)
y_real = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_real, pred_real))
r2 = r2_score(y_real, pred_real)

print("\n======================")
print("NEXT DAY MAN-HOUR")
print("======================")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12,5))
plt.plot(y_real, label="Actual")
plt.plot(pred_real, "--", label="Predicted")
plt.title("Next Day Man-Hour Forecast")
plt.xlabel("Day")
plt.ylabel("Man-Hour")
plt.grid()
plt.legend()
plt.show()

# ============================================================
# SAVE
# ============================================================

ask = int(input("Save Model? (0=No,1=Yes): "))

if ask == 1:
    model.save("best_manhour_nn.h5")
    joblib.dump(scaler_x, "scaler_x_manhour.pkl")
    joblib.dump(scaler_y, "scaler_y_manhour.pkl")
    print("Model Saved")