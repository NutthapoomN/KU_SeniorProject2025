# Forecast People Next 10 Minutes using LSTM


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
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

SEQ_LENGTH = 288

# ============================================================
# FUNCTION
# ============================================================

def encode_cyclical(df, col, max_val):
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def create_sequences_multistep(
    data,
    feature_cols,
    target_col,
    seq_length,
    forecast_horizon):

    xs = []
    ys = []

    feat = data[feature_cols].values
    targ = data[target_col].values

    for i in range(len(data)- seq_length- forecast_horizon+ 1):

        x = feat[i:i+seq_length]
        y = targ[i+seq_length:i+seq_length+forecast_horizon]

        xs.append(x)
        ys.append(y)

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
Data["People_Lag_144"] = Data["People"].shift(144)
Data["People_Lag_1008"] = Data["People"].shift(1008)
Data["People_RollMean_144"] = Data["People"].rolling(144).mean()
Data["People_RollMean_3"] = (Data["People"].rolling(3).mean())
Data["People_RollMean_6"] = (Data["People"].rolling(6).mean())

Data = Data.dropna().reset_index(drop=True)

# ============================================================
# LABEL
# ============================================================



Data = Data.dropna().reset_index(drop=True)

# ============================================================
# FEATURES
# ============================================================

features = [
    "DoW_sin","DoW_cos",
    "DoM_sin","DoM_cos",
    "Month_sin","Month_cos",
    "Hr_sin","Hr_cos",
    "Mn_sin","Mn_cos",
    "IsHoliday",
    "People",
    #"People_Lag_144",
    "People_Lag_1008",
    "People_RollMean_144"
]

target = "People"

# ============================================================
# SPLIT DATAFRAME FIRST
# ============================================================

split_index = int(len(Data) * 0.7)
train_df = Data.iloc[:split_index].copy()

# ============================================================
# SCALE
# ============================================================

scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaler_x.fit(train_df[features])

scaler_y.fit(train_df[[target]])

Data[features] = scaler_x.transform(Data[features])

Data[[target]] = scaler_y.transform(Data[[target]])

# ============================================================
# SEQUENCE
# ============================================================

FORECAST_HORIZON = 144

X, y = create_sequences_multistep(
    Data,
    features,
    "People",
    SEQ_LENGTH,
    FORECAST_HORIZON
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
    Bidirectional(LSTM(128,return_sequences=True),input_shape=(SEQ_LENGTH,len(features))),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=False )),
    Dropout(0.2),
    Dense(256,activation="relu"),
    Dense(128,activation="relu"),
    Dense(FORECAST_HORIZON)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="mse",
    metrics=["mae"]
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
    patience=5,
    min_lr=1e-6
)

# ============================================================
# TRAIN
# ============================================================

history = model.fit(X_train,y_train, validation_data=(X_val, y_val),
    epochs=300,
    batch_size=64,
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

y_test_real = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
r2 = r2_score(y_test_real, pred_real)

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

model.save("TonsorFlow_Bi-LSTM_Forcast144Point_allday.h5")
joblib.dump(scaler_x,"scaler_x_TonsorFlow_LSTM_Forcast144Point_allday.pkl")
joblib.dump(scaler_y,"scaler_y_TonsorFlow_LSTM_Forcast144Point_allday.pkl")
print("\nModel Saved name : TonsorFlow_Bi-LSTM_Forcast144Point_allday")

# ============================================================
# EXTERNAL TEST (4 DAYS COMBINED)
# ============================================================

print("\n")
print("=" * 80)
print("EXTERNAL TEST : COMBINED 4 DAYS")
print("=" * 80)

# ------------------------------------------------------------
# LOAD ALL SHEETS
# ------------------------------------------------------------

test_frames = []

for sheet in TEST_SHEETS:
    tmp = pd.read_excel(TEST_FILE,sheet_name=sheet)
    tmp["SourceSheet"] = sheet
    test_frames.append(tmp)
df = pd.concat(test_frames,ignore_index=True)
print(f"Total Rows : {len(df)}")

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

df = encode_cyclical(df, "Mn", 60)
df = encode_cyclical(df, "Hr", 24)
df = encode_cyclical(df, "DoW", 7)
df = encode_cyclical(df, "DoM", 31)
df = encode_cyclical(df, "Month", 12)

df["IsHoliday"] = df["IsHoliday"].astype(float)
df["Schedule"] = df["Schedule"].astype(float)
df["People"] = df["People"].astype(float)

df["People_Lag_1008"] = (df["People_Lag_1008"].astype(float))
df["People_RollMean_144"] = (df["People"].rolling(144).mean())


df = df.dropna().reset_index(drop=True)

# ------------------------------------------------------------
# SCALE
# ------------------------------------------------------------

df[features] = scaler_x.transform(df[features])
df[[target]] = scaler_y.transform(df[[target]])

# ------------------------------------------------------------
# CREATE SEQUENCE
# ------------------------------------------------------------

X_sheet, y_sheet = (
    create_sequences_multistep(
        df,
        features,
        target,
        SEQ_LENGTH,
        FORECAST_HORIZON
        ))

print("\nSequence Shape")
print("X :", X_sheet.shape)
print("y :", y_sheet.shape)

# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

pred = model.predict(X_sheet,verbose=0)
pred_real = scaler_y.inverse_transform(pred)
y_real = scaler_y.inverse_transform(y_sheet)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

mae = mean_absolute_error(y_real.flatten(),pred_real.flatten())
rmse = np.sqrt(mean_squared_error(y_real.flatten(),pred_real.flatten()))
r2 = r2_score(y_real.flatten(), pred_real.flatten())

print("\n======================")
print("24-HOUR FORECAST PERFORMANCE")
print("======================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ------------------------------------------------------------
# MAN-HOUR
# ------------------------------------------------------------

actual_manhour = (np.sum(y_real[0])* (10/60))
pred_manhour = (np.sum(pred_real[0]) * (10/60))
manhour_error = (pred_manhour- actual_manhour)
manhour_error_pct = (abs(manhour_error)/ max(actual_manhour, 1e-8)) * 100
actual_all = np.sum(y_real,axis=1)*(10/60)
pred_all = np.sum(pred_real,axis=1)*(10/60)

manhour_mae = mean_absolute_error(actual_all,pred_all)

print("\n======================")
print("MAN-HOUR ANALYSIS")
print("======================")
print(f"Actual Man-Hour    : {actual_manhour:.2f}")
print(f"Predicted Man-Hour : {pred_manhour:.2f}")
print(f"Error              : {manhour_error:.2f}")
print(f"Error (%)          : {manhour_error_pct:.2f}%")
print(f"Man-Hour MAE : {manhour_mae:.2f}")

# ------------------------------------------------------------
# PLOT FIRST 24-HOUR FORECAST
# ------------------------------------------------------------

time_axis = pd.date_range(start="00:00",  periods=FORECAST_HORIZON, freq="10min").strftime("%H:%M")

plt.figure(figsize=(18,6))

plt.plot(time_axis, y_real[0],label="Actual")
plt.plot(time_axis,pred_real[0],"--",label="Predicted")

plt.title("24 Hour Forecast")

plt.xlabel("Time")
plt.ylabel("People")

plt.xticks(np.arange( 0,FORECAST_HORIZON,12), rotation=45)

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# SAVE RESULT
# ------------------------------------------------------------

summary_df = pd.DataFrame([[
    mae,
    rmse,
    r2,
    actual_manhour,
    pred_manhour,
    manhour_error,
    manhour_error_pct
]], columns=[
    "MAE",
    "RMSE",
    "R2",
    "Actual_ManHour",
    "Predicted_ManHour",
    "ManHour_Error",
    "ManHour_Error_%"
])

print("\n")
print(summary_df)
ANSWER = int(input("Save model ? (Yes=1/No=0):"))
if ANSWER == 1 :
    summary_df.to_excel(r"D:\Senior Project\Test_Result_TonsorFlow_LSTM_Forcast144Point_allday.xlsx", index=False)

    print("\nSummary Saved")