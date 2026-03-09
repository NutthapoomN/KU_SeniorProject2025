import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==========================================================
# Persistent Fuzzy System (สร้างครั้งเดียว)
# ==========================================================
def create_fuzzy_system():

    # -------- INPUTS --------
    people = ctrl.Antecedent(np.arange(0, 32, 1), 'people')
    room_temp = ctrl.Antecedent(np.arange(20, 31, 1), 'room_temp')
    tes_temp = ctrl.Antecedent(np.arange(5, 31, 1), 'tes_temp')

    # -------- OUTPUTS --------
    cond_use = ctrl.Consequent(np.arange(0, 2.1, 0.1), 'cond_use')
    comp_use = ctrl.Consequent(np.arange(0, 2.1, 0.1), 'comp_use')
    pump_speed = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'pump_speed')
    fan_speed = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'fan_speed')
    valve_2 = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'valve_2')

    # -------- MEMBERSHIP FUNCTIONS --------
    people['no'] = fuzz.trapmf(people.universe, [0, 0, 0, 2])
    people['few'] = fuzz.trapmf(people.universe, [1, 1, 4, 6])
    people['medium'] = fuzz.trapmf(people.universe, [5, 10, 15, 15])
    people['many'] = fuzz.trapmf(people.universe, [14, 25, 31, 31])

    room_temp['normal'] = fuzz.trimf(room_temp.universe, [20 ,20, 26])
    room_temp['hot'] = fuzz.trimf(room_temp.universe, [24, 27, 29])
    room_temp['very_hot'] = fuzz.trapmf(room_temp.universe, [28, 29, 30, 30])

    tes_temp['off'] = fuzz.trimf(tes_temp.universe, [0, 0, 1])
    tes_temp['normal'] = fuzz.trimf(tes_temp.universe, [5, 10, 14])
    tes_temp['high'] = fuzz.trimf(tes_temp.universe, [13, 28, 30])

    for output in [cond_use, comp_use]:
        output['off'] = fuzz.trimf(output.universe, [0, 0, 0])
        output['on']  = fuzz.trimf(output.universe, [0, 0.3, 0.5])
        output['onmore']  = fuzz.trimf(output.universe, [0.4, 0.8, 1])

    fan_speed['off'] = fuzz.trimf(fan_speed.universe, [0, 0, 0])
    fan_speed['on'] = fuzz.trimf(fan_speed.universe, [0.5, 1, 1])
    fan_speed['onmore'] = fuzz.trimf(fan_speed.universe, [0.8, 2, 2])

    pump_speed['off'] = fuzz.trimf(pump_speed.universe, [0, 0, 0.5])
    pump_speed['on'] = fuzz.trimf(pump_speed.universe, [0.5, 1, 1])

    valve_2['-1_night'] = fuzz.trimf(valve_2.universe, [-1, -1, 0])
    valve_2['0_night'] = fuzz.trimf(valve_2.universe, [0, 0, 0.5])
    valve_2['1_day'] = fuzz.trimf(valve_2.universe, [0.5, 1, 1])


    # -------- RULES --------
    rules = [
        
        ctrl.Rule(people['no'] & room_temp['normal'] & tes_temp['off'],
                  [valve_2['1_day'], cond_use['off'], fan_speed['off'], comp_use['off'], pump_speed['off']]),
        ctrl.Rule(people['no'] & room_temp['hot'] & tes_temp['off'],
                  [valve_2['1_day'], cond_use['off'], fan_speed['off'], comp_use['off'], pump_speed['off']]),
        ctrl.Rule(people['no'] & room_temp['very_hot'] & tes_temp['off'],
                  [valve_2['1_day'], cond_use['off'], fan_speed['off'], comp_use['off'], pump_speed['off']]),  
        
        ctrl.Rule(people['few'] & room_temp['normal'] & tes_temp['normal'],
                  [valve_2['1_day'], cond_use['off'], fan_speed['on'], comp_use['off'], pump_speed['on']]),
        ctrl.Rule(people['few'] & room_temp['hot'] & tes_temp['normal'],
                  [valve_2['1_day'], cond_use['off'], fan_speed['onmore'], comp_use['off'], pump_speed['on']]),
        ctrl.Rule(people['few'] & room_temp['very_hot'] & tes_temp['normal'],
                  [valve_2['1_day'], cond_use['off'], fan_speed['onmore'], comp_use['off'], pump_speed['on']]),         
        ctrl.Rule(people['few'] & room_temp['normal'] & tes_temp['high'],
                  [valve_2['1_day'], cond_use['on'], fan_speed['on'], comp_use['on'], pump_speed['off']]),

        ctrl.Rule(people['few'] & room_temp['hot'] & tes_temp['high'],
                  [valve_2['1_day'], cond_use['onmore'], fan_speed['onmore'], comp_use['onmore'], pump_speed['off']]),

        ctrl.Rule(people['few'] & room_temp['very_hot'] & tes_temp['high'],
                  [valve_2['1_day'], cond_use['onmore'], fan_speed['onmore'], comp_use['onmore'], pump_speed['off']]),
        ctrl.Rule(people['medium'] & room_temp['normal'] & tes_temp['normal'],
                  [valve_2['1_day'], cond_use['on'], fan_speed['on'], comp_use['on'], pump_speed['on']]),                 
        ctrl.Rule(people['medium'] & room_temp['hot'] & tes_temp['normal'],
                  [valve_2['1_day'], cond_use['onmore'], fan_speed['onmore'], comp_use['onmore'], pump_speed['on']]),          
        ctrl.Rule(people['medium'] & room_temp['very_hot'] & tes_temp['normal'],
                  [valve_2['1_day'], cond_use['onmore'], fan_speed['onmore'], comp_use['onmore'], pump_speed['on']]),        

        ctrl.Rule(people['many'] & room_temp['normal'] & tes_temp['high'],
                  [valve_2['1_day'], cond_use['on'], fan_speed['on'], comp_use['on'], pump_speed['on']]),                              
        ctrl.Rule(people['many'] & room_temp['hot'] & tes_temp['high'],
                  [valve_2['1_day'], cond_use['onmore'], fan_speed['onmore'], comp_use['onmore'], pump_speed['on']]),          
        ctrl.Rule(people['many'] & room_temp['very_hot'] & tes_temp['high'],
                  [valve_2['1_day'], cond_use['onmore'], fan_speed['onmore'], comp_use['onmore'], pump_speed['on']]),          
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


# ==========================================================
# Controller (Time-Step Function)
# ==========================================================
fuzzy_sim = create_fuzzy_system()

def tes_controller(day, hour, people, t_room, t_tes, t_set):

    cond = 0
    v2 = 0
    pump = 0
    fan = 0
    comp = 0

    # ==============================
    # MODE 1 : Night Charging
    # ==============================
    if hour >= 22 or hour < 9:

        if people < 1:
            if t_tes > 5:
                return -1,1,0,1,0
            else:
                return 0,0,0,0,0
        else:
            if t_tes > 5 and t_room > t_set:
                return 0,1,1,1,0
            elif t_tes > 5 and t_room <= t_set:
                return -1,1,0,1,0
            elif t_tes <= 5 and t_room > t_set:
                return 0,1,1,1,0
            else:
                return 0,0,0,0,0


    # ==============================
    # MODE 2 : Day Operation
    # ==============================
    else:

        if t_room > t_set:

            fuzzy_sim.reset()

            fuzzy_sim.input['people'] = people
            fuzzy_sim.input['room_temp'] = t_room
            fuzzy_sim.input['tes_temp'] = t_tes

            try:
                fuzzy_sim.compute()
            except:
                return 0,0,0,0,0

            output = fuzzy_sim.output

            if len(output) == 0:
                return 0,0,0,0,0

            v2   = output.get('valve_2',0)
            cond = output.get('cond_use',0)
            comp = output.get('comp_use',0)
            fan  = output.get('fan_speed',0)
            pump = output.get('pump_speed',0)

            # normalize valve
            if v2 > 0:
                v2 = 1
            elif v2 < 0:
                v2 = -1
            else:
                v2 = 0

            if people < 1:
                return 0,0,0,0,0

            if t_tes > 14:
                pump = 0

            if cond < 0.1: cond = 0
            if comp < 0.1: comp = 0
            if fan < 0.1: fan = 0

            return v2,cond,fan,comp,pump

        else:
            return 0,0,0,0,0

#V2,Cond,Fan,Comp,Pump = tes_controller(1, 8, 12, 28, 25, 25)
#print(V2,Cond,Fan,Comp,Pump)




#Tester (D,hr,people,t_room,T_TES,T_set,expect)
#Output = V,fan cond ,fan evap, comp , pump 
class colors:
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m' 

T1 = [0,3,1,24,12,25,'Case 6']
T2 = [0,2,0,27,24,25,'Case 6']
T3 = [0,11,0,25,8,25,'Case 1']
T4 = [1,16,0,24,8,25,'Case 1']
T5 = [2,13,0,26,10,25,'Case 1']
T6 = [3,11,0,30,20,25,'Case 1']
T7 = [1,13,15,24,24,25,'Case 1']
T8 = [1,13,2,24,24,25,'Case 1']
T9 = [1,13,15,24,24,25,'Case 1']
T10 = [4,12,2,28,7,24,'Case 3']
T11 = [4,12,2,25,8,24,'Case 3']
T12 = [4,12,2,28,25,24,'Case 2'] #อาจจะ case 3
T13 = [3,12,10,30,7,24,'Case 4']
T14 = [3,12,12,29,7,24,'Case 4']
T15 = [3,12,13,29,7,24,'Case 4']
T16 = [5,3,5,24,12,25,'Case 6']
T17 = [6,7,10,26,7,25,'Case 5']
T18 = [0,3,0,29,6,25,'Case 6']
T19 = [0,3,0,27.5,12,25,'Case 6']
T20 = [0,3,0,27.5,12,25,'Case 6']
T21 = [5,3,0,23.5,15,25,'Case 6']
Tester = [T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20]

def Checkcase(V,fc,fe,comp,p):
    if fc == 0 and fe==0 and comp == 0 and p==0 : return 'Case 1'
    if V >0 and fc > 0 and fe>0 and comp > 0 and p==0 : return 'Case 2'
    if V >0 and fc == 0 and fe>0 and comp == 0 and p>0 : return 'Case 3'
    if V >0 and fc > 0 and fe>0 and comp > 0 and p>0 : return 'Case 4'
    if V ==0 and fc > 0 and fe>0 and comp > 0 and p==0 : return 'Case 5'
    if V ==0 and fc > 0 and fe==0 and comp > 0 and p==0 : return 'Case 5.1'
    if V <1 and fc >0 and fe==0 and comp > 0 and p==0 : return 'Case 6'
    else : return 'Case 0'


total_score = 0
stat = 0
a=[]
for i in (Tester):
    score=0
    s1,s2,s3,s4,s5,s6,T = float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4]),float(i[5]),i[6]
    ans1,ans2,ans3,ans4,ans5 = tes_controller(s1,s2,s3,s4,s5,s6)###########################################ใส่ฟังชัน
    C = Checkcase(round(float(ans1),2),round(float(ans2),2),round(float(ans3),2),round(float(ans4),2),round(float(ans5),2))
    ans = round(float(ans1),2),round(float(ans2),2),round(float(ans3),2),round(float(ans4),2),round(float(ans5),2)
    if C ==T:
        total_score +=1
        score=1
    stat += 1
    if score >0 :
        print(f'ทดสอบ T{stat} คือ {s1,s2,s3,s4,s5,s6} ได้ค่า {ans}        |เป็น  {C}  คาดหวัง {T} {colors.GREEN}score {score}{colors.ENDC}')
    else :
        print(f'ทดสอบ T{stat} คือ {s1,s2,s3,s4,s5,s6} ได้ค่า {ans}        |เป็น  {C}  คาดหวัง {T} {colors.FAIL}score {score}{colors.ENDC}')
print(f'รวมคะแนน {total_score}/{len(Tester)}')

