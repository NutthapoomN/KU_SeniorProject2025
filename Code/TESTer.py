import openpyxl

#Tester (D,hr,people,t_room,T_TES,T_set,expect)
#Output = V,fan cond ,fan evap, comp , pump 
set = []
for i in range(0,7): #Day
    for j in range(9,10): #Hr
        for k in range(0,32): #People
            for l in range(20,31): #T_Room
                for m in range(5,27) : #T_TES
                    for n in range(24,27): #T_set
                        #กลางคืน ไม่มีคน
                        if k <= 0 and (j>22 or j<9): N = 6
                        if k <= 0 and (j>22 or j<9) and m == 5: N = 1
                        #กลางคืน มีคน
                        if k > 0 and (j>22 or j<9): N = 5
                        if k > 0 and (j>22 or j<9) and l==m : N = 5.1
                        #กลางวัน ไม่มีคน
                        if k <= 0 and not((j>22 or j<9)) : N = 1
                        #กลางวัน มีคน
                        if k > 1 and not((j>22 or j<9))and k<=6 and m<(0.4*(n-m))  : N = 34
                        if k > 1 and not((j>22 or j<9))and k<=6 and m>(0.4*(n-m))  : N = 2
                        if k > 6 and not((j>22 or j<9))and k<=15 and m<(0.4*(n-m))  : N = 34
                        if k > 6 and not((j>22 or j<9))and k<=15 and m>(0.4*(n-m))  : N = 234
                        if k > 15 and not((j>22 or j<9))and k<=31 and m<(0.4*(n-m))  : N = 234
                        if k > 15 and not((j>22 or j<9))and k<=1315 and m>(0.4*(n-m))  : N = 34
                        Ex = str(N)
                        data = [i,j,k,l,m,n,Ex]
                        set.append(data)
print(len(set))


#Tester (D,hr,people,t_room,T_TES,T_set,expect)
#Output = V,fan cond ,fan evap, comp , pump 
class colors:
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m' 
def Checkcase(V,fc,fe,comp,p):
    if fc == 0 and fe==0 and comp == 0 and p==0 : return '1'
    if V >0 and fc > 0 and fe>0 and comp > 0 and p==0 : return '2'
    if V >0 and fc == 0 and fe>0 and comp == 0 and p>0 : return '3'
    if V >0 and fc > 0 and fe>0 and comp > 0 and p>0 : return '4'
    if V ==0 and fc > 0 and fe>0 and comp > 0 and p==0 : return '5'
    if V ==0 and fc > 0 and fe==0 and comp > 0 and p==0 : return '5.1'
    if V <1 and fc >0 and fe==0 and comp > 0 and p==0 : return '6'
    else : return '0'

total_score = 0
stat = 0
a=[]
for i in (set):
    score=0
    s1,s2,s3,s4,s5,s6,T = float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4]),float(i[5]),i[6]
    ans1,ans2,ans3,ans4,ans5 = tes_controller(s1,s2,s3,s4,s5,s6)
    C = Checkcase(round(float(ans1),2),round(float(ans2),2),round(float(ans3),2),round(float(ans4),2),round(float(ans5),2))
    ans = round(float(ans1),2),round(float(ans2),2),round(float(ans3),2),round(float(ans4),2),round(float(ans5),2)
    if C in T:
        total_score +=1
        score=1
    stat += 1
print(f'รวมคะแนน {colors.GREEN}{total_score}{colors.ENDC}/{len(set)} ( {round(total_score/len(set),3)} % )')