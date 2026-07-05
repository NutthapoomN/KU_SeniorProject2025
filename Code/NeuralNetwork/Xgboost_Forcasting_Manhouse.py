import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# =====================================================
# CONFIG
# =====================================================

DATA_FILE = r"D:\Senior Project\Data\Data_Train_3_Year.xlsx"
SHEET_NAME = "Train_manhouse"

FEATURE = [
    "DoW",
    "IsHoliday",
    "Term",
    "Schedule",
    #"ManHour_Lag1",
    #"ManHour_Lag7"
]

TARGET = "Man-House"

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel(
    DATA_FILE,
    sheet_name=SHEET_NAME
)

df = df.dropna().reset_index(drop=True)

print("Rows :", len(df))

# =====================================================
# TRAIN / VAL / TEST
# =====================================================

train_end = int(len(df)*0.70)
val_end = int(len(df)*0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

X_train = train_df[FEATURE]
y_train = train_df[TARGET]

X_val = val_df[FEATURE]
y_val = val_df[TARGET]

X_test = test_df[FEATURE]
y_test = test_df[TARGET]

# =====================================================
# MODEL
# =====================================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train,y_train),
        (X_val,y_val)
    ],
    verbose=False
)

# =====================================================
# PREDICT
# =====================================================

pred = model.predict(X_test)

# =====================================================
# METRICS
# =====================================================

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)

print("\n======================")
print("XGBOOST RESULT")
print("======================")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature":FEATURE,
    "Importance":model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(14,5))

plt.plot(y_test.values,label="Actual")
plt.plot(pred, "--", label="Predicted")

plt.title("Next Day Man-Hour Forecast (XGBoost)")

plt.xlabel("Day")
plt.ylabel("Man-Hour")

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# SAVE
# =====================================================

ASK_SAVE = int(input("Save Model? (0=No , 1=Yes): "))

if ASK_SAVE == 1:
    joblib.dump(model,"best_xgboost_manhour.pkl")
    print("Model Saved")
else:
    print("No Save")