"""import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model('TEST_Version_LSTM_for_MPC.h5')
scaler_X = joblib.load('TEST_Version_scaler_X.pkl')
scaler_Y = joblib.load('TEST_Version_scaler_y.pkl')

# =====================================================================
# 2. ดึงข้อมูลดิบจาก Excel
# =====================================================================
df = pd.read_excel(r'D:\Senior Project\Matlab&Simulink\Train_Scenario1.xlsx', sheet_name="Scenario1")

# รายชื่อคอลัมน์ทั้ง 19 ตัวที่คุณกำหนดไว้ (Features สำหรับ X)
col = ['f_Evap', 'f_Cond', 'f_Compsr', 'f_Pump', 'f_v_Master', 'f_v_Charge', 'f_v_Discharge', 
       'People', 'T_Envi', 'T_Room', 'T_Tank', 'T_Supply', 'T_Return', 'SOC', 
       'Q_Cond', 'Q_Evap', 'Q_Charge', 'Q_Discharge', 'Q_Load']

# ดึงข้อมูลมา 120 แถวแรกเพื่อสร้างเป็น 1 ชุดข้อมูล (SeqLength = 120)
seq_length = 120
x_raw = df[col].head(seq_length).values  # ได้ Matrix ขนาด 2 มิติ (120, 19)

# =====================================================================
# 3. กระบวนการ Rescale และปรับมิติให้เป็น 3D (สำหรับ Input X)
# =====================================================================
# ขั้นแรก: ทำการ Scale ข้อมูลดิบในรูปแบบ 2 มิติก่อน
x_scaled_2d = scaler_X.transform(x_raw)

# ขั้นสอง: เพิ่มมิติ (Expand Dimension) ให้กลายเป็น 3 มิติ เพื่อส่งให้ LSTM
# จาก (20, 19) จะกลายเป็น (1, 20, 19) -> [Batch=1, SeqLength=20, Features=19]
x_input_3d = np.expand_dims(x_scaled_2d, axis=0)

# =====================================================================
# 4. ส่งข้อมูลให้โมเดลพยากรณ์ผล (Predict)
# =====================================================================
# ค่าที่ได้ออกมาจากตรงนี้ จะยังเป็นค่าที่ถูกสเกลอยู่ (Scaled Predict) มิติจะเป็น (1, 3)
y_pred_scaled = model.predict(x_input_3d)

# =====================================================================
# 5. แปลงค่าทำนายกลับเป็นค่าจริง (Inverse Transform) และคืนค่าออกมา
# =====================================================================
# นำค่าที่โมเดลทายได้มาแปลงกลับด้วย scaler_Y เพื่อให้ได้ค่าในหน่วยปกติ
y_final = scaler_Y.inverse_transform(y_pred_scaled)

# แยกผลลัพธ์ออกมาเป็นตัวแปรตามลำดับ Target ของคุณ
t_room_pred = y_final[0][0]
t_tank_pred = y_final[0][1]
soc_pred = y_final[0][2]

# แสดงผลลัพธ์สุดท้ายที่ได้
print("\n================ ผลการพยากรณ์ ณ เวลา (T+1) ================")
print(f"T_Room(T+1)  : {t_room_pred:.4f} °C")
print(f"T_Tank(T+1)  : {t_tank_pred:.4f} °C")
print(f"SOC(T+1)     : {soc_pred:.4f} %")
print("==========================================================")"""
import sys
import tensorflow as tf
import keras

print("--- ตรวจสอบเวอร์ชันในเครื่องปัจจุบัน ---")
print(f"Python Version     : {sys.version}")
print(f"TensorFlow Version : {tf.__version__}")
print(f"Keras Version      : {keras.__version__}")
print("------------------------------------\n")