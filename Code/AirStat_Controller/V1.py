import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------- Load Data ----------
df = pd.read_excel('Code/AirStat_Controller/Class_Schedule_AirStat.xlsx')
Feature = ['DayOfWeek','Time','User']
X = df[Feature].values.astype('float32')
Y = df['AIR_STAT'].values.astype('int64')  # class 0/1

# ---------- Split ----------
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# ---------- MinMax Scale 0–1 ----------
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)   # scaled -> 0..1
x_test = sc.transform(x_test)

# save scaler
joblib.dump(sc, "V1_Air_Controller_MinMax.save")

# Convert to tensor
x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(x_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# ---------- PyTorch Model ----------
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

model = AirNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- Train ----------
epochs = 50
for epoch in range(epochs):
    for xb, yb in train_dl:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

# ---------- Evaluate ----------
with torch.no_grad():
    test_logits = model(x_test_t)
    test_pred = test_logits.argmax(1)
    accuracy = (test_pred == y_test_t).float().mean().item()

print("\nTest accuracy:", accuracy * 100, "%")

# ---------- Export Model with ArgMax (0 or 1) ----------
class AirNetArgMax(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits = self.net(x)
        pred = torch.argmax(logits, dim=1)
        return pred  # result is 0 or 1
        

model_argmax = AirNetArgMax(model)

example_input = torch.randn(1, 3)
traced_argmax = torch.jit.trace(model_argmax, example_input)

traced_argmax.save("V1_Air_Controller_for_Simulink_ArgMax2.pt")

print("✅ Saved ArgMax version: V1_Air_Controller_for_Simulink_ArgMax.pt")
# ---------- PRINT TEST SCALED + PRINT SCALE VALUES ----------
print("\n===== X_TEST AFTER SCALING (0-1) =====")
print(x_test_t)

print("\n===== MinMax Scale Values for MATLAB =====")
print("Min = ", sc.data_min_)
print("Max = ", sc.data_max_)
print("Scale = ", sc.scale_)   # 1/(max-min)
print("Min_Offset = ", sc.min_)  # -min/(max-min)

