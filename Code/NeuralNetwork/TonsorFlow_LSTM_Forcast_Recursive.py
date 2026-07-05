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
    epochs=50,
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
# EXTERNAL TEST (DAY-BY-DAY RECURSIVE FORECAST)
# ============================================================

summary_result = []

sheet_pairs = [
    ("21-12-2025", "22-12-2025"),
    ("22-12-2025", "23-12-2025"),
    ("23-12-2025", "24-12-2025"),
    ("24-12-2025", "25-12-2025")
]

for history_sheet, target_sheet in sheet_pairs:

    print("\n")
    print("=" * 80)
    print(f"HISTORY : {history_sheet}")
    print(f"TARGET  : {target_sheet}")
    print("=" * 80)

    # ========================================================
    # LOAD DATA
    # ========================================================

    history_df = pd.read_excel(
        TEST_FILE,
        sheet_name=history_sheet
    )

    target_df = pd.read_excel(
        TEST_FILE,
        sheet_name=target_sheet
    )

    history_len = len(history_df)

    df = pd.concat(
        [history_df, target_df],
        ignore_index=True
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = encode_cyclical(df, "Mn", 60)
    df = encode_cyclical(df, "Hr", 24)
    df = encode_cyclical(df, "DoW", 7)
    df = encode_cyclical(df, "DoM", 31)
    df = encode_cyclical(df, "Month", 12)

    df["IsHoliday"] = df["IsHoliday"].astype(float)
    df["Schedule"] = df["Schedule"].astype(float)

    df["People"] = df["People"].astype(float)

    if "People_Lag_1008" in df.columns:
        df["People_Lag_1008"] = df["People_Lag_1008"].astype(float)

    # ========================================================
    # MIXED COLUMN
    # ========================================================

    df["People_Mixed"] = df["People"].copy()

    forecast_start_idx = history_len

    print(f"Forecast starts at index : {forecast_start_idx}")
    print(f"Forecast rows            : {len(target_df)}")

    # ========================================================
    # RECURSIVE FORECAST
    # ========================================================

    for i in range(forecast_start_idx, len(df)):

        # -------------------------------
        # Rolling Mean from PAST ONLY
        # -------------------------------

        past_df = df.iloc[max(0, i-6):i]

        if len(past_df) >= 3:
            roll3 = past_df["People_Mixed"].tail(3).mean()
        else:
            roll3 = past_df["People_Mixed"].mean()

        if len(past_df) >= 6:
            roll6 = past_df["People_Mixed"].tail(6).mean()
        else:
            roll6 = past_df["People_Mixed"].mean()

        df.loc[i, "People_RollMean_3"] = roll3
        df.loc[i, "People_RollMean_6"] = roll6

        # -------------------------------
        # Build Sequence
        # -------------------------------

        seq_df = df.iloc[i-SEQ_LENGTH:i].copy()

        seq_df["People"] = seq_df["People_Mixed"]

        # Rebuild rolling feature
        seq_df["People_RollMean_3"] = (
            seq_df["People_Mixed"]
            .rolling(3)
            .mean()
            .bfill()
        )

        seq_df["People_RollMean_6"] = (
            seq_df["People_Mixed"]
            .rolling(6)
            .mean()
            .bfill()
        )

        # -------------------------------
        # Scale
        # -------------------------------

        seq_scaled = scaler_x.transform(
            seq_df[features]
        )

        X_input = np.expand_dims(
            seq_scaled,
            axis=0
        )

        # -------------------------------
        # Predict
        # -------------------------------

        pred_scaled = model.predict(
            X_input,
            verbose=0
        )

        pred_val = scaler_y.inverse_transform(
            pred_scaled
        )[0][0]

        pred_val = max(0, pred_val)

        df.loc[i, "People_Mixed"] = pred_val

    # ========================================================
    # EVALUATION
    # ========================================================

    eval_df = df.iloc[forecast_start_idx:].copy()

    y_real = eval_df["People"].values.reshape(-1, 1)

    pred_real = eval_df["People_Mixed"].values.reshape(-1, 1)

    mae = mean_absolute_error(
        y_real,
        pred_real
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            pred_real
        )
    )

    r2 = r2_score(
        y_real,
        pred_real
    )

    mape = np.mean(
        np.abs(
            (y_real - pred_real)
            /
            np.clip(
                y_real,
                1e-8,
                None
            )
        )
    ) * 100

    acc = 100 - mape

    # ========================================================
    # MAN-HOUR
    # ========================================================

    actual_manhour = np.sum(y_real) * (10/60)

    pred_manhour = np.sum(pred_real) * (10/60)

    manhour_error = (
        pred_manhour
        -
        actual_manhour
    )

    manhour_error_pct = (
        abs(manhour_error)
        /
        max(actual_manhour, 1e-8)
    ) * 100

    print("\n----- RESULT -----")

    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"MAPE  : {mape:.2f}%")
    print(f"ACC   : {acc:.2f}%")

    print(f"\nActual ManHour : {actual_manhour:.2f}")
    print(f"Pred ManHour   : {pred_manhour:.2f}")
    print(f"Error          : {manhour_error:.2f}")
    print(f"Error (%)      : {manhour_error_pct:.2f}%")

    summary_result.append([
        target_sheet,
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

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(14,5))

    plt.plot(
        y_real,
        label="Actual"
    )

    plt.plot(
        pred_real,
        "--",
        label="Recursive Forecast"
    )

    plt.title(
        f"{target_sheet} Recursive Forecast"
    )

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
        "ManHour_Error_%"
    ]
)

print(summary_df)

summary_df.to_excel(
    r"D:\Senior Project\Data\Test_Result.xlsx",
    index=False
)

print("\nSummary Saved")