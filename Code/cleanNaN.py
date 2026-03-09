import pandas as pd
import numpy as np

# 1. ระบุชื่อไฟล์ของคุณ
file_path = r'D:\Senior Project\Data\22-12-2025.xlsx'

try:
    # อ่านไฟล์ Excel (ระบุ sheet_name ถ้าไม่ใช่หน้าแรก)
    df = pd.read_excel(file_path, sheet_name='22-12-2025')
    
    print(f"--- กำลังตรวจสอบไฟล์: {file_path} ---")
    
    # 2. ค้นหา NaN (ค่าว่าง)
    nan_locations = np.where(pd.isnull(df))
    nan_coords = list(zip(nan_locations[0] + 2, nan_locations[1] + 1)) # +2 เพราะ Excel เริ่มแถว 1 และมี Header
    
    # 3. ค้นหา Inf (ค่าอนันต์)
    # เฉพาะคอลัมน์ที่เป็นตัวเลขเท่านั้น
    df_numeric = df.select_dtypes(include=[np.number])
    inf_locations = np.where(np.isinf(df_numeric))
    inf_coords = list(zip(inf_locations[0] + 2, inf_locations[1] + 1))

    # 4. แสดงผลลัพธ์
    if not nan_coords and not inf_coords:
        print("✅ ไม่พบค่า NaN หรือ Inf ในไฟล์นี้")
    else:
        if nan_coords:
            print(f"❌ พบ NaN (ค่าว่าง) จำนวน {len(nan_coords)} จุด ที่พิกัด (แถว, คอลัมน์):")
            print(nan_coords)
            
        if inf_coords:
            print(f"❌ พบ Inf (ค่าอนันต์) จำนวน {len(inf_coords)} จุด ที่พิกัด (แถว, คอลัมน์):")
            print(inf_coords)

    # 5. ตรวจสอบข้อมูลที่ไม่ใช่ตัวเลขในคอลัมน์ที่ควรเป็นตัวเลข (Data Type Mismatch)
    print("\n--- ตรวจสอบ Data Type ---")
    print(df.dtypes)

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")