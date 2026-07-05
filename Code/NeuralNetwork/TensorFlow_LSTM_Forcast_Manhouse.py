# Forecast Man-Hour Next Day using LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = r"D:\Senior Project\Data\Data_Train_3_Year.xlsx"
SHEET_NAME = "Train_manhouse"

FEATURE = ["DoW","DoM","Month","Schedule","IsHoliday","Term","ManHour_Lag7"]
TARGET = "Man-House"

SEQ_LENGTH = 7

# ============================================================
# FUNCTION
# ============================================================

def create_sequences(data, feature_cols, target_col, seq_length):
    X, y = [], []
    feat = data[feature_cols].values
    targ = data[target_col].values

    for i in range(len(data)-seq_length):
        X.append(feat[i:i+seq_length])
        y.append(targ[i+seq_length])

    return np.array(X), np.array(y)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
#df = df.dropna().reset_index(drop=True)

print("Rows :", len(df))

# ============================================================
# SPLIT BEFORE SCALE
# ============================================================

split_index = int(len(df)*0.7)
train_df = df.iloc[:split_index].copy()

# ============================================================
# SCALE
# ============================================================

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(train_df[FEATURE])
scaler_y.fit(train_df[[TARGET]])

df[FEATURE] = scaler_x.transform(df[FEATURE])
df[[TARGET]] = scaler_y.transform(df[[TARGET]])

# ============================================================
# CREATE SEQUENCE
# ============================================================

X, y = create_sequences(df, FEATURE, TARGET, SEQ_LENGTH)

print("X Shape :", X.shape)
print("y Shape :", y.shape)

# ============================================================
# TRAIN / VAL / TEST
# ============================================================

train_size = int(len(X)*0.7)
val_size = int(len(X)*0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# ============================================================
# MODEL
# ============================================================

model = Sequential([
    Bidirectional(LSTM(128), input_shape=(SEQ_LENGTH, len(FEATURE))),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dense(1, activation="relu")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    shuffle=False,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()

# ============================================================
# TEST PERFORMANCE
# ============================================================

pred = model.predict(X_test, verbose=0)

pred_real = scaler_y.inverse_transform(pred)
y_real = scaler_y.inverse_transform(y_test.reshape(-1,1))

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
# PLOT RESULT
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
# SAVE MODEL
# ============================================================
ASK_SAVE = int(input("Save Model? (0 = No  , 1 =Yes) :"))
if ASK_SAVE == 1 :
    model.save("best_manhour_model.h5")
    joblib.dump(scaler_x, "scaler_x_manhour.pkl")
    joblib.dump(scaler_y, "scaler_y_manhour.pkl")

    print("\nModel Saved")
else : print("-- No save --")