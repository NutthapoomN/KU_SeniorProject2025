import matplotlib.pyplot as plt

def Entry_room(row,Time,Number):
    Hr,Mn,Sec ='','',''
    n=0
    for i in Time:
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


dd = '15'   # ใส่วันที่บันทึกข้อมูล
# row = [Time, Date, Hr, Mn, Sec, Entry, Exit, NumberPeople, Light]
row =[]
Time = 3600*24

C_Sec=0
C_MN=0
C_Hr=0
for i in range(1,Time+1):
    C_Sec = i
    if C_Sec%60==0 and C_Sec!=0: 
        C_Sec =0
        C_MN+=1
    if C_MN%60 ==0 and C_MN!=0:
        C_MN=0
        C_Hr+=1
    row.append([i,'2025-12-'+dd,C_Hr,C_MN,C_Sec,0,0,0])




## กรอกข้อมูลตรงนี้

Entry_room(row,'10:00:00',1)
Entry_room(row,'10:01:00',3)
Entry_room(row,'10:02:00',3)
Entry_room(row,'12:03:00',13)
Entry_room(row,'13:04:00',10)
Entry_room(row,'10:10:00',3)
Entry_room(row,'10:11:00',3)
Entry_room(row,'10:12:00',3)
Entry_room(row,'10:13:00',3)
Entry_room(row,'10:14:00',-3)
Entry_room(row,'10:15:00',-5)
Entry_room(row,'10:20:00',-5)
Entry_room(row,'11:16:10',10)
Entry_room(row,'12:30:00',-13)



X,Y1,Y2 =[],[],[]
for i in row:
    X.append(i[0])
    Y1.append(i[5])
    Y2.append(i[7])

#plt.plot(X,Y1,label='Entry', color='Black')
plt.plot(X,Y2,label='In Room', color='red')
plt.legend()
plt.show()
