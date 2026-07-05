
f_Evap = out.f_Evap.Data;
f_Cond = out.f_Cond.Data;
f_Compsr = out.f_Compsr.Data;
f_Pump = out.f_Pump.Data;
f_v_Master = out.f_v_Master.Data;
f_v_Charge = out.f_v_Charge.Data;
f_v_Discharge = out.f_v_Discharge.Data;
People = out.n_People.Data;
T_Envi = out.T_Envi.Data;
T_Room = out.T_Room.Data; 
T_Tank = out.T_Tank.Data;
T_Supply = out.T_Supply.Data;
T_Return = out.T_Return.Data;
SOC = out.SOC.Data;
Q_Cond = out.Q_Cond.Data;
Q_Evap = out.Q_Evap.Data;
Q_Charge = out.Q_Charge.Data;
Q_Discharge = out.Q_Discharge.Data;
Q_Load = out.Q_Load.Data;

% 2. แปลงข้อมูล Timeseries เป็น MATLAB Table

T = table(t, ...
    f_Evap,f_Cond, f_Compsr, f_Pump,f_v_Master, f_v_Charge, f_v_Discharge, ...
    People,T_Envi,T_Room,T_Tank, T_Supply, T_Return, SOC,Q_Cond, Q_Evap, ...
    Q_Charge,Q_Discharge,Q_Load,'VariableNames', { ...
    'Time_s','f_Evap','f_Cond','f_Compsr','f_Pump', 'f_v_Master','f_v_Charge', 'f_v_Discharge', ...
    'People','T_Envi','T_Room','T_Tank','T_Supply','T_Return', 'SOC','Q_Cond', 'Q_Evap', ...
    'Q_Charge','Q_Discharge','Q_Load'});

% 3. เขียนข้อมูลลงไฟล์ Excel
filename = 'Train_Scenario5.xlsx';
writetable(T, filename, 'Sheet', 'Scenario5');

disp('บันทึกข้อมูลลง Excel เรียบร้อยแล้ว!');