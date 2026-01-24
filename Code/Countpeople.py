import matplotlib.pyplot as plt
def Read_Number(Text):
    Hr,Mn,Sec ='','',''
    n=0
    for i in Text:
        if i ==':' : 
            n+=1
            continue
        if n == 0 :
            Hr += i
        elif n== 1 :
            Mn +=i
        elif n==2 :
            Sec += i
    Hr =int(Hr)
    Mn = int(Mn)
    Sec = int(Sec)
    print(Hr,Mn,Sec)
    return Hr,Mn,Sec
def Entry_room(row,Time,Number):
    Hr,Mn,Sec=Read_Number(Time)
    for i in row :
        if i[2] == Hr and i[3]== Mn and i[4] == Sec:
            print(f'เวลา {Hr,Mn,Sec} จะเพิ่มจำนวน {Number}')
            if Number <=0 :
                i[6]=Number
            else :
                i[5]=Number
            print(f'แก้ไขแล้วได้ {i}')
    print(Hr,Mn,Sec,'Success')
    print(i)
    Count = 0
    for i in row:
        Count =i[5] +Count + i[6]
        i[7]= Count
    return 
def IsDoorOpen(row,TimeOpen,TimeClosed):
    H_O,M_O,S_O = Read_Number(TimeOpen)
    H_C,M_C,S_C= Read_Number(TimeClosed)
    print(H_O,M_O,S_O)
    print(H_C,M_C,S_C)
    print("เริ่ม loop")
    Stat = 0
    for i in row :
        if i[2] == H_C and i[3]==M_C and i[4] == S_C:
            Stat = 0
            print(f"แก้ไขจนถึงเวลา : {i[2],i[3],i[4]}")
            return
        if i[2] == H_O and i[3]==M_O and i[4] == S_O:
            Stat = 1
            print(f'เริ่มแก้ไขที่ {i[2],i[3],i[4]}')
        if Stat == 1 :
            Stat += 1
        if Stat ==2 :
            i[9] =1
            print(i)
    return
def IsLightOpen(row,Time,Stat):
    Hr,Mn,Sec=Read_Number(Time)
    if Stat == 'Open':
        V = 1
    else :
        V=0
    print('เจอ Start')
    Start=0
    for i in row :
            if i[2] == Hr and i[3]== Mn and i[4] == Sec:
                Start = 1
            if Start==1:
                i[8]=V
                print('Change')
    return
def IsAirOpen(row,Time,Stat):
    Hr,Mn,Sec=Read_Number(Time)
    if Stat == 'Open':
        V = 1
    else :
        V=0
    print('เจอ Start')
    Start=0
    for i in row :
            if i[2] == Hr and i[3]== Mn and i[4] == Sec:
                Start = 1
            if Start==1:
                i[10]=V
                print('Change')
    return

dd = '22'   # ใส่วันที่บันทึกข้อมูล
# row = [0Time, 1Date, 2Hr, 3Mn, 4Sec, 5Entry, 6Exit, 7NumberPeople, 8Light, 9Door, 10Air]
row =[]
Time = 3600*24

C_Sec=0
C_MN=0
C_Hr=0
for i in range(1,Time+1):
    C_Sec+=1
    if C_Sec%60==0 and C_Sec!=0: 
        C_Sec =0
        C_MN+=1
    if C_MN%60 ==0 and C_MN!=0:
        C_MN=0
        C_Hr+=1
    row.append([i,'2025-12-'+dd,C_Hr,C_MN,C_Sec,0,0,0,0,0,0])



## กรอกข้อมูลตรงนี้ 22/12/2025 ##
Entry_room(row,'07:20:16',1)
IsDoorOpen(row,'07:20:14','07:20:19')
IsLightOpen(row,'07:20:21','Open')
Entry_room(row,'07:24:06',-1)
IsDoorOpen(row,'07:24:05','07:24:08')
Entry_room(row,'07:24:26',1)
IsDoorOpen(row,'07:24:26','07:24:28')
IsLightOpen(row,'07:24:32','Close')
Entry_room(row,'07:24:26',-1)
IsDoorOpen(row,'07:24:37','07:24:41')

Entry_room(row,'10:06:34',1)
IsDoorOpen(row,'10:06:33','10:06:37')
IsLightOpen(row,'10:06:41','Open')
IsLightOpen(row,'10:06:47','Close')
Entry_room(row,'10:06:51',-1)
IsDoorOpen(row,'10:06:49','10:06:54')
IsDoorOpen(row,'10:15:42','10:15:47')
Entry_room(row,'10:15:43',1)
IsLightOpen(row,'10:15:51','Open')
IsDoorOpen(row,'10:16:40','10:16:55')
Entry_room(row,'10:16:53',1)
IsDoorOpen(row,'10:17:14','10:17:50')
Entry_room(row,'10:17:14',1)
Entry_room(row,'10:17:50',-1)
IsDoorOpen(row,'10:23:10','10:23:14')
Entry_room(row,'10:23:11',-1)
IsDoorOpen(row,'10:33:04','10:33:07')
Entry_room(row,'10:33:04',1)
IsDoorOpen(row,'10:45:20','10:45:26')

IsDoorOpen(row,'10:59:24','10:59:28')
Entry_room(row,'10:59:25',1)
IsDoorOpen(row,'11:00:00','11:00:06')
Entry_room(row,'11:00:01',-1)

IsDoorOpen(row,'11:39:08','11:39:14')
Entry_room(row,'11:39:10',1)
IsDoorOpen(row,'11:41:09','11:41:15')
Entry_room(row,'11:41:10',1)
IsDoorOpen(row,'12:24:02','12:24:25')
Entry_room(row,'12:24:18',-2)
Entry_room(row,'12:24:19',-1)

Entry_room(row,'12:45:09',1)
IsDoorOpen(row,'12:45:08','12:45:11')
Entry_room(row,'12:46:04',1)
IsDoorOpen(row,'12:46:04','12:46:07')
Entry_room(row,'12:50:07',1)
IsDoorOpen(row,'12:50:07','12:50:12')
Entry_room(row,'12:50:30',-1)
IsDoorOpen(row,'12:50:29','12:50:34')
Entry_room(row,'12:52:30',1)
IsDoorOpen(row,'12:52:13','12:52:17')
Entry_room(row,'12:52:43',-1)
IsDoorOpen(row,'12:52:43','12:52:47')
Entry_room(row,'12:52:53',1)
IsDoorOpen(row,'12:52:53','12:52:59')
Entry_room(row,'12:53:45',1)
IsDoorOpen(row,'12:53:45','12:53:51')
Entry_room(row,'12:54:02',1)
Entry_room(row,'12:54:03',1)
IsDoorOpen(row,'12:54:01','12:54:06')
Entry_room(row,'12:54:08',1)
IsDoorOpen(row,'12:54:07','12:54:13')
Entry_room(row,'12:54:28',1)
IsDoorOpen(row,'12:54:28','12:54:32')
Entry_room(row,'12:55:47',1)
IsDoorOpen(row,'12:55:46','12:55:52')
Entry_room(row,'12:56:54',2)
Entry_room(row,'12:56:56',2)
Entry_room(row,'12:56:57',1)
IsDoorOpen(row,'12:56:42','12:57:01')
Entry_room(row,'12:56:57',-1)
IsDoorOpen(row,'12:57:13','12:57:18')
Entry_room(row,'12:58:05',-1)
Entry_room(row,'12:58:07',-1)
IsDoorOpen(row,'12:58:03','12:58:13')
Entry_room(row,'12:58:20',-1)
IsDoorOpen(row,'12:58:18','12:58:26')
Entry_room(row,'12:58:59',1)
IsDoorOpen(row,'12:58:56','12:59:00')
Entry_room(row,'12:59:31',1)
IsDoorOpen(row,'12:59:31','12:59:37')
Entry_room(row,'12:59:45',1)
IsDoorOpen(row,'12:59:44','12:59:48')
Entry_room(row,'13:00:22',1)
IsDoorOpen(row,'13:00:21','13:00:24')
Entry_room(row,'13:03:56',-1)
IsDoorOpen(row,'13:03:53','13:03:59')
Entry_room(row,'13:04:51',-1)
IsDoorOpen(row,'13:04:51','13:04:57')
Entry_room(row,'13:05:17',1)
IsDoorOpen(row,'13:05:17','13:05:22')
Entry_room(row,'13:05:56',1)
IsDoorOpen(row,'13:05:52','13:05:59')
Entry_room(row,'13:06:37',1)
IsDoorOpen(row,'13:06:37','13:06:42')
Entry_room(row,'13:07:57',1)
IsDoorOpen(row,'13:07:57','13:08:08')
Entry_room(row,'13:08:39',1)
Entry_room(row,'13:08:43',1)
Entry_room(row,'13:08:44',1)
Entry_room(row,'13:08:45',1)
IsDoorOpen(row,'13:08:37','13:08:48')
Entry_room(row,'13:08:53',-1)
IsDoorOpen(row,'13:08:53','13:08:58')
Entry_room(row,'13:09:04',1)
IsDoorOpen(row,'13:09:04','13:09:09')
Entry_room(row,'13:11:04',1)
IsDoorOpen(row,'13:11:37','13:11:43')
Entry_room(row,'13:17:03',1)
IsDoorOpen(row,'13:17:00','13:17:06')
Entry_room(row,'13:18:23',-1)
IsDoorOpen(row,'13:18:22','13:18:26')
Entry_room(row,'13:20:53',1)
IsDoorOpen(row,'13:20:53','13:20:56')
Entry_room(row,'13:31:10',1)
IsDoorOpen(row,'13:31:10','13:31:16')
Entry_room(row,'13:34:28',1)
IsDoorOpen(row,'13:34:24','13:34:30')
Entry_room(row,'13:39:42',-1)
IsDoorOpen(row,'13:39:42','13:39:47')
Entry_room(row,'13:51:09',1)
IsDoorOpen(row,'13:51:08','13:51:13')
Entry_room(row,'13:58:31',1)
IsDoorOpen(row,'13:58:30','13:58:36')
Entry_room(row,'14:14:16',1)
IsDoorOpen(row,'14:14:15','14:14:26')
Entry_room(row,'14:14:20',1)
Entry_room(row,'14:14:22',2)
Entry_room(row,'14:15:58',-1)
IsDoorOpen(row,'14:15:57','14:16:01')
Entry_room(row,'14:17:54',1)
IsDoorOpen(row,'14:17:53','14:17:58')
Entry_room(row,'14:28:39',1)
Entry_room(row,'14:28:47',-1)
IsDoorOpen(row,'14:28:38','14:28:52')
Entry_room(row,'14:30:18',-1)
IsDoorOpen(row,'14:30:18','14:30:22')
Entry_room(row,'14:36:52',-1)
IsDoorOpen(row,'14:36:51','14:36:56')
Entry_room(row,'14:38:16',-1)
IsDoorOpen(row,'14:38:16','14:38:21')
Entry_room(row,'14:39:02',-1)
IsDoorOpen(row,'14:38:58','14:39:04')
Entry_room(row,'14:44:12',-1)
IsDoorOpen(row,'14:44:12','14:44:17')
Entry_room(row,'14:46:20',1)
IsDoorOpen(row,'14:46:20','14:46:25')
Entry_room(row,'14:50:05',-1)
IsDoorOpen(row,'14:50:05','14:50:10')
Entry_room(row,'14:51:46',1)
IsDoorOpen(row,'14:51:46','14:51:52')
IsDoorOpen(row,'14:55:25','14:55:40')
Entry_room(row,'15:00:00',-1)
IsDoorOpen(row,'14:59:56','15:00:01')
Entry_room(row,'15:02:36',1)
IsDoorOpen(row,'15:02:36','15:02:43')
Entry_room(row,'15:10:26',-1)
IsDoorOpen(row,'15:10:26','15:02:28')
Entry_room(row,'15:11:12',-1)
IsDoorOpen(row,'15:11:10','15:11:10')
IsDoorOpen(row,'15:14:02','15:14:22')
Entry_room(row,'15:14:10',-2)
IsDoorOpen(row,'15:14:40','15:15:04')
Entry_room(row,'15:14:51',-1)
Entry_room(row,'15:14:53',-2)
Entry_room(row,'15:15:46',-5)
Entry_room(row,'15:15:50',-4)
IsDoorOpen(row,'15:15:46','15:32:23')
Entry_room(row,'15:16:07',-1)
Entry_room(row,'15:16:10',-1)
Entry_room(row,'15:17:00',-2)
Entry_room(row,'15:32:22',-2)
Entry_room(row,'15:09:30',-1)
IsDoorOpen(row,'16:08:48','16:09:31')
print('---------------------------------')
IsLightOpen(row,'16:09:55','Close')
IsDoorOpen(row,'16:09:59','16:10:05')
Entry_room(row,'16:09:59',-3)


'''nnnn= 16*3600+9*60+50
for i in row:
    if i[1] == nnnn:
        print(i)
    elif i[0] < nnnn+10:
        print(i)
    else:
        break'''


x=[]
y=[]
for i in row:
    x.append(i[0])
    y.append(i[8])
plt.plot(x, y)

# ตั้งชื่อแกน
plt.xlabel("x")
plt.ylabel("y")

# ตั้งชื่อกราฟ
plt.title("Example X-Y Plot")

# แสดงกราฟ
plt.show()

print('เหลือ : ',row[-1][7],' คน')

# ข้อมูลหัวข้อ
import pandas as pd
columns = ['Time', 'Date', 'Hour', 'Minute', 'Second', 'Entry', 'Exit', 'NumberPeople', 'Light', 'Door','AirStat']
df1 = pd.DataFrame(row, columns=columns)

# ใช้ ExcelWriter เพื่อเขียนหลาย Sheet ลงในไฟล์เดียว
with pd.ExcelWriter(r'From_Camera2.xlsx') as writer:
    df1.to_excel(writer, sheet_name='22-12-2025', index=False)

    
print("สร้างไฟล์ Excel แล้ว")
