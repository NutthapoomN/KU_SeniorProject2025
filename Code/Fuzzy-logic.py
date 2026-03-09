import time
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import datetime


# 1. ส่วนพยากรณ์ (Advanced Prediction)
# ตัวอย่างข้อมูลย้อนหลัง 7 วัน (Index 0-6 คือ จันทร์-อาทิตย์)
# แต่ละ List มี 24ค่า นับ 0-23
history_people_data = {
    0: [0,0,10,50,50,20,50,10,0,0,0,0,10,50,50,20,50,10,0,0,10,20,15,0], # จันทร์
    1: [0,0,0,10,40,40,40,10,0,0,0,0,10,40,40,40,20,40,10,0,0,10,0,0], # อังคาร
    2: [0,0,10,40,40,20,40,10,0,0,0,10,40,40,20,40,10,0,0,0,0,10,30,0], # พุธ
    3: [0,0,10,40,40,40,20,40,10,0,0,0,10,50,50,20,50,10,0,0,0,10,5,10], # พฤหัส
    4: [0,0,0,10,40,20,40,10,0,0,0,0,10,40,40,20,40,10,0,0,10,20,0,0], # ศุกร์
    5: [0,10,40,40,20,40,10,0,0,00,0,0,10,40,20,40,10,0,0,10,0,0,0], # เสาร์
    6: [0,0,0,5,10,5,5,5,0,0,0,0,0,5,10,5,5,5,0,0,10,15,0,0]  # อาทิตย์
}


def get_historical_prediction():
    # 1. ดูว่าวันนี้วันอะไร (0=จันทร์, 6=อาทิตย์)
    now = datetime.datetime.now()
    day_of_week = now.weekday()
    daily_people = history_people_data.get(day_of_week, [0]*24) 
   
    predicted_people = []
    for i in range(1,11):
        future_time = now + datetime.timedelta(minutes=i * 10)
        future_hour = future_time.hour
        
        # ดึงค่าคนจากข้อมูลรายชั่วโมง (ใช้ % 24 เพื่อให้วนกลับมาตอนเที่ยงคืนได้)
        val = daily_people[future_hour % 24]
        predicted_people.append(val)

    return predicted_people


def get_charge_target(hourly_people, hourly_ambient_temp, hourly_room_temp):
    """คำนวณโหลดความร้อนโดยเทียบส่วนต่างอุณหภูมิระหว่าง 'นอกห้อง' กับ 'ในห้อง' จริงๆ"""
    total_load = 0
    
    for i in range(len(hourly_people)):
        # 1. โหลดจากคน (Internal Gain)
        h_people = (hourly_people[i] * 100 )/6
        # 2. โหลดความร้อนไหลเข้า (Heat Gain from Envelope)
        if hourly_ambient_temp[i] > hourly_room_temp[i]:
            delta_t = hourly_ambient_temp[i] - hourly_room_temp[i]
            h_weather = (delta_t * 500)/6  # 500 คือค่า UA (Heat Transfer Coefficient * Area)
        else:
            h_weather = 0 
        total_load += (h_people + h_weather)
    
    max_cap = 50000  # ค่าสมมติความจุสูงสุดของถัง
    target = (total_load / max_cap) * 100
    return max(20, min(100, target)) #ยังไงก็ต้องชาร์จชั้นต่ำ20% ถึงพรุ่งนี้จะไม่มีโหลดเลยก็ตาม





# 2. ส่วนสมอง Fuzzy Logic (The Decision Maker)
def setup_fuzzy_logic():
    # กำหนด Input
    people = ctrl.Antecedent(np.arange(0, 51, 1), 'people')
    room_temp = ctrl.Antecedent(np.arange(20, 41, 1), 'room_temp')
    
    # กำหนด Output
    tes_use = ctrl.Consequent(np.arange(0, 101, 1), 'tes_use')
    comp_use = ctrl.Consequent(np.arange(0, 101, 1), 'comp_use')
    pump_speed = ctrl.Consequent(np.arange(0, 101, 1), 'pump_speed')

    # Membership Functions
    people['few'] = fuzz.trapmf(people.universe, [0, 0, 5, 10])
    people['many'] = fuzz.trapmf(people.universe, [15, 30, 50, 50])
    room_temp['cool'] = fuzz.trimf(room_temp.universe, [20, 20, 26])
    room_temp['hot'] = fuzz.trapmf(room_temp.universe, [27, 32, 40, 40])
    
    tes_use['high'] = fuzz.trimf(tes_use.universe, [40, 100, 100])
    comp_use['off'] = fuzz.trimf(comp_use.universe, [0, 0, 10])
    comp_use['on'] = fuzz.trimf(comp_use.universe, [40, 100, 100])
    pump_speed['low'] = fuzz.trimf(pump_speed.universe, [0, 0, 50])
    pump_speed['high'] = fuzz.trimf(pump_speed.universe, [40, 100, 100])

    # กฎการเลือกโหมด (ต้องเพิ่มอีกหลายกรณี)
    rule1 = ctrl.Rule(people['few'] & room_temp['hot'], [tes_use['high'], comp_use['off']]) # คนน้อย -> TES 100%
    rule2 = ctrl.Rule(people['many'] & room_temp['hot'], [tes_use['high'], comp_use['on']]) # คนมาก -> Hybrid
    rule3 = ctrl.Rule(people['few'] & room_temp['cool'], [tes_use['high'], comp_use['off']])
    rule4 = ctrl.Rule(people['many'] & room_temp['cool'], [tes_use['high'], comp_use['on']])

    
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3, rule4]))




# ==========================================
# 3. ส่วนควบคุม Hardware (ตามไดอะแกรมวาล์ว 1-5)
# ==========================================
def update_hardware(cond1, v2, v3, pump4, fan5, comp6):
    print(f"\n[STATUS] Cond:{cond1} V2:{v2} V3:{v3} Pump:{pump4} Fan:{fan5} Comp:{comp6}")



# ==========================================
# 4. ส่วนระบบหลัก (Main Loop)
# ==========================================
def run_system():

    fuzzy_sim = setup_fuzzy_logic() 
    now = datetime.datetime.now()
    h = now.hour 
    # # มาจาก import time จะอ่านค่าเวลาในคอมตอนนั้น
    hour = [6,7, 8, 9, 10, 19, 23]

    # ดึงค่าพยากรณ์คนจากการคำนวณสถิติ 7 วันย้อนหลัง 
    p_data = get_historical_prediction() 
    
    # อุณหภูมิพยากรณ์
    t_ambient = [26, 27, 28, 30, 32, 35, 36, 35, 34, 28]
    t_room = [25, 25, 25, 26, 27, 26, 25, 25, 25, 24]

    # ส่งข้อมูล 100 นาทีไปคำนวณเป้าหมายการชาร์จ TES
    charge_target = get_charge_target(p_data, t_ambient, t_room)
    
    print(f"\n--- เวลาปัจจุบัน {now.strftime('%H:%M')} (Day {now.weekday()}) ---")
    print(f"ทำนายล่วงหน้า 100 นาที: {p_data}")
    print(f"เป้าหมายการชาร์จ TES: {charge_target:.2f}%")


    for h in hour:
        print(f"\n--- กำลังทดสอบเวลา {h}:00 น. ---")
        # --- MODE 1: ชาร์จตอนดึก (22:00 - 09:00) ---
        if h >= 22 or h < 9:
            print(f"Mode: Charging TES (Target: {charge_target:.1f}%)")
            update_hardware(1, 0, 1, 0, 1, 1)

        # --- MODE 2: ใช้งาน (09:00 - 22:00) ---
        elif 9 <= h < 22:
            # สมมติค่าจาก Sensor จริง
            curr_people = 30 
            curr_temp = 32
            
            fuzzy_sim.input['people'] = curr_people
            fuzzy_sim.input['room_temp'] = curr_temp
            fuzzy_sim.compute()
            
            tes_p = fuzzy_sim.output['tes_use']
            comp_p = fuzzy_sim.output['comp_use']

            if curr_people > 15:
                print(f"Mode: Hybrid (TES {tes_p:.1f}% + Comp {comp_p:.1f}%)")
                update_hardware(1, 1, 0, 1, 1, 1)
            else: 
                print(f"Mode: TES Only (TES {tes_p:.1f}% + Comp {comp_p:.1f}%)")
                update_hardware(0, 1, 0, 1, 1, 0, 1)

        # --- MODE 3: ปิดระบบ ---
        else:
            print("Mode: System Off")
            update_hardware(0, 0, 0, 0, 0, 0, 0)

        print("\n[INFO] รออีก 10 นาทีเพื่อตรวจสอบรอบถัดไป")
        time.sleep(5) # หน่วงเวลา 10 นาที

if __name__ == "__main__":
    run_system()