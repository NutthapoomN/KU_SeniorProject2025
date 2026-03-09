import openpyxl

#Tester (D,hr,people,t_room,T_TES,T_set,expect)
#Output = V,fan cond ,fan evap, comp , pump 
set = []
for i in range(0,7): #Day
    for j in range(0,24): #Hr
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
print(set[-1])