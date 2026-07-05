# Forecast People Next 10 Minutes using LSTM


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.layers import Bidirectional, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = r"D:\Senior Project\Data\Data_Train_3_Year.xlsx"

TRAIN_SHEETS = [
    "Year_2023",
    "Year_2024",
    "Year_2025"
]

TEST_FILE = r"D:\Senior Project\Data\Camera_DetectPeople_10mn_Version.xlsx"

TEST_SHEETS = [
    "22-12-2025",
    "23-12-2025",
    "24-12-2025",
    "25-12-2025"
]

SEQ_LENGTH = 36

# ============================================================
# FUNCTION
# ============================================================

def encode_cyclical(df, col, max_val):
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def create_sequences(data,feature_cols,target_col,seq_length):
    xs = []
    ys = []
    data_feat = data[feature_cols].values
    data_target = data[target_col].values
    for i in range(len(data) - seq_length):
        xs.append(data_feat[i:i + seq_length])
        ys.append(data_target[i + seq_length])
    return np.array(xs), np.array(ys)


# ============================================================
# LOAD TRAIN DATA
# ============================================================

frames = []

for sheet in TRAIN_SHEETS:
    df = pd.read_excel(TRAIN_FILE, sheet_name=sheet)
    print(f"Loaded {sheet} : {len(df)} rows")
    frames.append(df)
Data = pd.concat(frames,ignore_index=True)
Data = Data.dropna().reset_index(drop=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

Data = encode_cyclical(Data, "Mn", 60)
Data = encode_cyclical(Data, "Hr", 24)
Data = encode_cyclical(Data, "DoW", 7)
Data = encode_cyclical(Data, "DoM", 31)
Data = encode_cyclical(Data, "Month", 12)

Data["IsHoliday"] = Data["IsHoliday"].astype(float)
Data["Schedule"] = Data["Schedule"].astype(float)
Data["People"] = Data["People"].astype(float)
Data["People_Lag_1008"] = Data["People"].shift(1008)
Data["People_RollMean_3"] = (Data["People"].rolling(3).mean())
Data["People_RollMean_6"] = (Data["People"].rolling(6).mean())

Data = Data.dropna().reset_index(drop=True)

# ============================================================
# LABEL
# ============================================================

Data["People_Next10"] = (Data["People"].shift(-1))

Data = Data.dropna().reset_index(drop=True)

# ============================================================
# FEATURES
# ============================================================

features = [
    "DoW_sin", "DoW_cos",
    "DoM_sin", "DoM_cos",
    "Month_sin", "Month_cos",
    "Hr_sin", "Hr_cos",
    "Mn_sin", "Mn_cos",
    "IsHoliday",
    "Schedule",
    "People",
    "People_Lag_1008",
    "People_RollMean_3",
    "People_RollMean_6"
]

target = "People_Next10"

# ============================================================
# SPLIT DATAFRAME FIRST
# ============================================================

split_index = int(len(Data) * 0.7)
train_df = Data.iloc[:split_index].copy()

# ============================================================
# SCALE
# ============================================================

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(train_df[features])

scaler_y.fit(train_df[[target]])

Data[features] = scaler_x.transform(Data[features])

Data[[target]] = scaler_y.transform(Data[[target]])

# ============================================================
# SEQUENCE
# ============================================================

X, y = create_sequences(
    Data,
    features,
    target,
    SEQ_LENGTH
)

# ============================================================
# TRAIN / VAL / TEST
# ============================================================

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size: train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

# ============================================================
# MODEL
# ============================================================

model = Sequential([
    Bidirectional(LSTM(128), input_shape=(SEQ_LENGTH, len(features))),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
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

history = model.fit(X_train,y_train, validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# TRAIN CURVE
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"],label="Validation")

plt.legend()
plt.grid()
plt.show()

# ============================================================
# TEST SET PERFORMANCE
# ============================================================

pred = model.predict(X_test,verbose=0)

pred_real = scaler_y.inverse_transform(pred)

y_test_real = scaler_y.inverse_transform(y_test.reshape(-1,1))

mae = mean_absolute_error(y_test_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
r2 = r2_score(y_test_real, pred_real)
mape = np.mean(np.abs((y_test_real - pred_real)/np.clip(y_test_real,1e-8, None))) * 100

print("\n======================")
print("TEST SET PERFORMANCE")
print("======================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
#print(f"MAPE : {mape:.2f}%")

# ============================================================
# SAVE MODEL
# ============================================================

model.save("best_AI_TES_Model_tf.h5")

joblib.dump(
    scaler_x,
    "scaler_x.pkl"
)

joblib.dump(
    scaler_y,
    "scaler_y.pkl"
)

print("\nModel Saved")

# ============================================================
# EXTERNAL TEST
# ============================================================

summary_result = []

for sheet in TEST_SHEETS:

    print("\n")
    print("=" * 80)
    print(sheet)
    print("=" * 80)

    df = pd.read_excel(
        TEST_FILE,
        sheet_name=sheet
    )

    df = encode_cyclical(df, "Mn", 60)
    df = encode_cyclical(df, "Hr", 24)
    df = encode_cyclical(df, "DoW", 7)
    df = encode_cyclical(df, "DoM", 31)
    df = encode_cyclical(df, "Month", 12)

    df["IsHoliday"] = df["IsHoliday"].astype(float)
    df["Schedule"] = df["Schedule"].astype(float)
    df["People"] = df["People"].astype(float)
    df["People_Lag_1008"] = df["People_Lag_1008"].astype(float)
    df["People_RollMean_3"] = df["People"].rolling(3).mean()
    df["People_RollMean_6"] = df["People"].rolling(6).mean()


    df = df.dropna()

    df["People_Next10"] = (df["People"].shift(-1))

    df = df.dropna()

    df[features] = scaler_x.transform(df[features])
    df[[target]] = scaler_y.transform(df[[target]])

    X_sheet, y_sheet = (create_sequences(df, features, target, SEQ_LENGTH))

    pred = model.predict(X_sheet, verbose=0)

    pred_real = scaler_y.inverse_transform(pred)

    y_real = scaler_y.inverse_transform(y_sheet.reshape(-1,1))



# ==========================================
# MAN-HOUR ANALYSIS
# ==========================================

    actual_manhour = np.sum(y_real) * (10/60)

    pred_manhour = np.sum(pred_real) * (10/60)

    manhour_error = pred_manhour - actual_manhour

    manhour_error_pct = (
        abs(manhour_error)
        / max(actual_manhour,1e-8)
    ) * 100

    print("\n----- MAN-HOUR -----")
    print(f"Actual Man-Hour    : {actual_manhour:.2f}")
    print(f"Predicted Man-Hour : {pred_manhour:.2f}")
    print(f"Error              : {manhour_error:.2f}")
    print(f"Error (%)          : {manhour_error_pct:.2f}%")

    mae = mean_absolute_error(y_real, pred_real )

    rmse = np.sqrt(mean_squared_error(y_real,pred_real))
    r2 = r2_score( y_real, pred_real )
    mape = np.mean(np.abs((y_real - pred_real) /np.clip(y_real, 1e-8,None))) * 100
    acc = 100 - mape

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    summary_result.append([
        sheet,
        mae,
        rmse,
        r2,
        mape,
        acc,
        actual_manhour,
        pred_manhour,
        manhour_error,
        manhour_error_pct
    ])

    plt.figure(figsize=(14,5))

    plt.plot(y_real[:150], label="Actual")
    plt.plot(pred_real[:150], "--", label="Predicted")

    plt.title(sheet)
    plt.legend()
    plt.grid()
    plt.show()

# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(

    summary_result,

    columns=[
        "Sheet",
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
        "Accuracy",
        "Actual_ManHour",
        "Predicted_ManHour",
        "ManHour_Error",
        "ManHour_Error_%"]
)

print("\n")
print(summary_df)
ANSWER = int(input("Save model ? (Yes=1/No=0):"))
if ANSWER == 1 :
    summary_df.to_excel(
        r"D:\Senior Project\Data\Test_Result.xlsx",
        index=False
    )

    print("\nSummary Saved")