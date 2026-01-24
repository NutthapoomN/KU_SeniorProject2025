import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import random

# ======================================================
#                PART 1 — CLASSIFICATION 0/1
# ======================================================

# ---------- Load Data ----------
df = pd.read_excel('Code/AirStat_Controller/Class_Schedule_AirStat.xlsx')
Feature = ['DayOfWeek','Time','User']
X = df[Feature].values.astype('float32')
Y = df['AIR_STAT'].values.astype('int64')  # class 0/1

# ---------- Split ----------
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# ---------- MinMax Scale ----------
sc1 = MinMaxScaler()
x_train = sc1.fit_transform(x_train)
x_test = sc1.transform(x_test)

joblib.dump(sc1, "V1_Air_Controller_MinMax.save")

# Convert to tensor
x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(x_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)


# ---------- Classification Model ----------
class AirNet(nn.Module):
    def __init__(self):
        super(AirNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # 2 classes

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model_cls = AirNet()
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.001)

# ---------- Train Classification ----------
epochs = 50
for epoch in range(epochs):
    for xb, yb in train_dl:
        optimizer_cls.zero_grad()
        preds = model_cls(xb)
        loss = criterion_cls(preds, yb)
        loss.backward()
        optimizer_cls.step()

    if epoch % 10 == 0:
        print(f"[CLS] Epoch {epoch} Loss {loss.item():.4f}")

# ---------- Evaluate ----------
with torch.no_grad():
    test_logits = model_cls(x_test_t)
    test_pred = test_logits.argmax(1)
    accuracy = (test_pred == y_test_t).float().mean().item()

print("\nClassification accuracy:", accuracy * 100, "%")

# ---------- Export ArgMax Model ----------
class AirNetArgMax(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits = self.net(x)
        pred = torch.argmax(logits, dim=1)
        return pred

model_argmax = AirNetArgMax(model_cls)
example_input = torch.randn(1, 3)
torch.jit.trace(model_argmax, example_input).save(
    "V1_Air_Controller_for_Simulink_ArgMax.pt"
)

print("✅ Saved: V1_Air_Controller_for_Simulink_ArgMax.pt")

# ======================================================
#         PART 2 — COOLING TIME PREDICTOR (REGRESSION)
# ======================================================

# Example dataset columns:
# DayOfWeek, Time, T_start, T_target, t_used
cool_df = pd.read_excel("Code/AirStat_Controller/CoolingData.xlsx")

cool_features = ['DayOfWeek','Time','T_start','T_target']
X2 = cool_df[cool_features].values.astype('float32')
Y2 = cool_df['t_used'].values.astype('float32').reshape(-1,1)

# ---------- Scale ----------
sc2 = MinMaxScaler()
X2_scaled = sc2.fit_transform(X2)

joblib.dump(sc2, "CoolTime_MinMaxScaler.save")

x2_t = torch.tensor(X2_scaled, dtype=torch.float32)
y2_t = torch.tensor(Y2, dtype=torch.float32)

ds2 = TensorDataset(x2_t, y2_t)
dl2 = DataLoader(ds2, batch_size=32, shuffle=True)

# ---------- Regression Model ----------
class CoolTimeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model_reg = CoolTimeNet()
criterion_reg = nn.MSELoss()
optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.001)

# ---------- Train ----------
epochs = 80
for epoch in range(epochs):
    for xs, ys in dl2:
        optimizer_reg.zero_grad()
        out = model_reg(xs)
        loss = criterion_reg(out, ys)
        loss.backward()
        optimizer_reg.step()
    if epoch % 10 == 0:
        print(f"[REG] Epoch {epoch} Loss {loss.item():.4f}")

torch.save(model_reg.state_dict(), "CoolTime_Model.pt")
print("✅ Saved: CoolTime_Model.pt")

# ======================================================
#      PART 3 — ONLINE LEARNING (FEEDBACK UPDATE)
# ======================================================

# Replay buffer to avoid forgetting
replay_buffer = []

def update_cooltime_online(x_raw, t_real):
    """
    x_raw = [DayOfWeek, Time, T_start, T_target]
    t_real = เวลาใช้จริงหลังชาร์จสำเร็จ
    """
    global model_reg, optimizer_reg, replay_buffer

    # scale input
    x_scaled = sc2.transform([x_raw])
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # store to buffer
    replay_buffer.append((x_tensor, torch.tensor([[t_real]])))
    if len(replay_buffer) > 2000:
        replay_buffer.pop(0)

    # sample mini batch
    batch = random.sample(replay_buffer, k=min(32, len(replay_buffer)))
    bx = torch.cat([b[0] for b in batch])
    by = torch.cat([b[1] for b in batch])

    # train
    model_reg.train()
    optimizer_reg.zero_grad()
    pred = model_reg(bx)
    loss = criterion_reg(pred, by)
    loss.backward()
    optimizer_reg.step()

    # save updated
    torch.save(model_reg.state_dict(), "CoolTime_Model_Updated.pt")

    print(f"🔁 Online update: loss={loss.item():.4f}")


# ======================================================
#    PART 4 — LOGIC: ชาร์จต้องเสร็จก่อน 9 โมงเช้า
# ======================================================

def should_start_charging(time_now, predicted_hours):
    """
    time_now    = เวลาปัจจุบัน (ชั่วโมงในรูปแบบ float เช่น 6.5 = 06:30)
    predicted_hours = เวลา cooling ที่ทำนายได้
    ต้องให้เสร็จ <= 9.0
    """
    finish_time = time_now + predicted_hours
    return finish_time <= 9.0


print("\n===== SYSTEM READY =====")
print("✔ Classification model loaded")
print("✔ Cooling-time regression model ready")
print("✔ Online learning enabled")
