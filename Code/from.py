# ============================================================
# Forecast People_Next10 using LSTM (PyTorch)
# Updated Features:
# DoW, DoM, Hr, Mn, Month, IsHoliday, Schedule, People
# Label:
# People_Next10
# ============================================================
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

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

# Cyclical Encoding
Data = encode_cyclical(Data, 'Mn', 60)
Data = encode_cyclical(Data, 'Hr', 24)
Data = encode_cyclical(Data, 'DoW', 7)
Data = encode_cyclical(Data, 'DoM', 31)
Data = encode_cyclical(Data, 'Month', 12)

# Convert Numeric
Data['IsHoliday'] = Data['IsHoliday'].astype(float)
Data['Schedule'] = Data['Schedule'].astype(float)
Data['People'] = Data['People'].astype(float)
# ============================================================
# 2.5 Lag Features (10,20,30 นาที)
# ============================================================
Data['Lag_10'] = Data['People'].shift(1)
Data['Lag_20'] = Data['People'].shift(2)
Data['Lag_30'] = Data['People'].shift(3)

# ============================================================
# 2.6 Rolling Mean
# ============================================================
Data['People_RollMean_3'] = Data['People'].rolling(3).mean()   # ~30 นาที
Data['People_RollMean_6'] = Data['People'].rolling(6).mean()   # ~1 ชั่วโมง

# ลบ NaN ที่เกิดจาก lag + rolling
Data = Data.dropna().reset_index(drop=True)
# ============================================================
# 3. Create Future Label
# ============================================================
Data['People_Next10'] = Data['People'].shift(-10)

# Remove NaN from shifted rows
Data = Data.dropna().reset_index(drop=True)

# ============================================================
# 4. Feature Selection
# ============================================================
features = [
    'DoW_sin','DoW_cos',
    'Hr_sin','Hr_cos',
    'Mn_sin','Mn_cos',
    'IsHoliday',
    'Schedule',
    'Lag_10',
    'Lag_20',
    'Lag_30',
    'People_RollMean_3',
    'People'
]

target = 'People_Next10'

# ============================================================
# 5. Scaling
# ============================================================
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

Data[features] = scaler_x.fit_transform(Data[features])
Data[[target]] = scaler_y.fit_transform(Data[[target]])

# ============================================================
# 6. Sequence Generator
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

SEQ_LENGTH = 18

X, y = create_sequences(
    Data,
    features,
    target,
    SEQ_LENGTH
)

# ============================================================
# 7. Train/Test Split
# ============================================================
train_size = int(len(X) * 0.7)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_loader = DataLoader(
    TensorDataset(
        torch.Tensor(X_train),
        torch.Tensor(y_train).view(-1, 1)
    ),
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(
        torch.Tensor(X_test),
        torch.Tensor(y_test).view(-1, 1)
    ),
    batch_size=32,
    shuffle=False
)

# ============================================================
# 8. LSTM Model
# ============================================================
class AI_TES_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AI_TES_Model, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
            )

        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout Layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        # LSTM Output
        out, _ = self.lstm(x)

        # ใช้ timestep สุดท้าย
        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # Fully Connected
        out = self.fc(out)

        return out


# ============================================================
# Model Initialization
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AI_TES_Model(
    input_dim=len(features),
    hidden_dim=64,
    num_layers=2,
    output_dim=1
).to(device)

# ============================================================
# 10. Training Setup
# ============================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',     # ดู validation loss
    factor=0.5,     # ลด LR ครึ่งหนึ่ง
    patience=5      # ถ้าไม่ดีขึ้น 5 epoch ค่อยลด
)
epochs = 500
patience = 15

best_v_loss = float('inf')
counter = 0

train_losses = []
test_losses = []

print(f"\nเริ่มการเทรนบนอุปกรณ์: {device}")

# ============================================================
# 11. Training Loop
# ============================================================
for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):

    # ======================
    # Training Phase
    # ======================
    model.train()
    t_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        t_loss += loss.item()

    # ======================
    # Validation Phase
    # ======================
    model.eval()
    v_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            v_loss += loss.item()

    avg_t = t_loss / len(train_loader)
    avg_v = v_loss / len(test_loader)
    scheduler.step(avg_v)
    train_losses.append(avg_t)
    test_losses.append(avg_v)

    # ======================
    # Update Progress Bar Info
    # ======================
    tqdm.write(
        f"Epoch {epoch+1:03d}/{epochs} | "
        f"Train Loss: {avg_t:.6f} | "
        f"Validation Loss: {avg_v:.6f}"
    )
    # ======================
    # Early Stopping
    # ======================
    if avg_v < best_v_loss:
        best_v_loss = avg_v
        torch.save(model.state_dict(), 'best_AI_TES_Model.pt')
        counter = 0
    else:
        counter += 1

        if counter >= patience:
            print(f"\n-- Early Stopping ที่ Epoch {epoch+1} --")
            break

# ============================================================
# 12. Visualization
# ============================================================
plt.figure(figsize=(14, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Validation Loss', color='red')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Prediction Plot
model.load_state_dict(torch.load('best_AI_TES_Model.pt'))
model.eval()

with torch.no_grad():
    sample_x = torch.Tensor(X_test).to(device)
    pred = model(sample_x).cpu().numpy()

# Reverse Scaling
pred_real = scaler_y.inverse_transform(pred)
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))

plt.subplot(1, 2, 2)

plt.plot(
    y_test_real[:150],
    label='Actual People_Next10',
    linewidth=1.5
)

plt.plot(
    pred_real[:150],
    label='Predicted People_Next10',
    linestyle='--',
    linewidth=1.5
)

plt.title('Actual vs Predicted (First 150 points)')
plt.xlabel('Time Steps')
plt.ylabel('People Count')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''
# ============================================================
# 13. Save Model & Scalers
# ============================================================
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

print("\n--- เสร็จสิ้นกระบวนการเทรนและประเมินผล ---")
print(f"โมเดลถูกบันทึกไว้ที่: {os.getcwd()}\\best_AI_TES_Model.pt")
print(f"Scaler X ถูกบันทึกไว้ที่: {os.getcwd()}\\scaler_x.pkl")
print(f"Scaler Y ถูกบันทึกไว้ที่: {os.getcwd()}\\scaler_y.pkl")'''

# ============================================================
# 12. Final Evaluation Metrics
# ============================================================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model.load_state_dict(torch.load('best_AI_TES_Model.pt'))
model.eval()

with torch.no_grad():
    sample_x = torch.Tensor(X_test).to(device)
    pred = model(sample_x).cpu().numpy()

# Reverse scaling
pred_real = scaler_y.inverse_transform(pred)
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# ============================================================
# Metrics
# ============================================================
mae = mean_absolute_error(y_test_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
r2 = r2_score(y_test_real, pred_real)

# MAPE (%)
mape = np.mean(
    np.abs((y_test_real - pred_real) / np.clip(y_test_real, 1e-8, None))
) * 100

# Accuracy approximation
accuracy = 100 - mape

print("\n================ Final Model Performance ================")
print(f"MAE   : {mae:.4f} คน")
print(f"RMSE  : {rmse:.4f} คน")
print(f"R²    : {r2:.4f}")
print(f"MAPE  : {mape:.2f}%")
print(f"Accuracy Approximation : {accuracy:.2f}%")
print("========================================================")

# ============================================================
# Optional Sample Comparison
# ============================================================
print("\nตัวอย่างผลทำนาย 10 ค่าแรก:")
for i in range(10):
    actual = y_test_real[i][0]
    predicted = pred_real[i][0]
    error = abs(actual - predicted)

    print(
        f"Sample {i+1:02d} | "
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f} | "
        f"Error: {error:.2f}"
    )