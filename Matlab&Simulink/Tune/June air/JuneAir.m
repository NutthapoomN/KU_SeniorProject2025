clc
clear
T = readtable('D:\Senior Project\Data\Excel\20_1_2026.xlsx');
Time = T{3190:5902,1};
JuneAir_T_0 = 0:1:2712;
JuneAir_T_0 = JuneAir_T_0.';
JuneAir_1_Voltage = T{3190:5902,8};
JuneAir_2_Frequency = T{3190:5902,9};
JuneAir_3_pow_gshp_ref_ac = T{3190:5902,10};
JuneAir_4_pow_gshp_gshp_ac = T{3190:5902,11};
JuneAir_5_pow_gshp_ref_cdu_compsr = T{3190:5902,12};
JuneAir_6_pow_gshp_ref_cdu_fan = T{3190:5902,13};
JuneAir_7_pow_gshp_gshp_cdu_fan = T{3190:5902,14};
JuneAir_8_pow_gshp_gshp_vfd = T{3190:5902,15};
JuneAir_9_pow_gshp_gshp_cdu_compsr= T{3190:5902,16};
JuneAir_10_pow_compos_compsr = T{3190:5902,17};
JuneAir_11_f_gshp_water = T{3190:5902,18};
JuneAir_12_f_gshp_r32 = T{3190:5902,19};
JuneAir_13_t_gshp_hex_in_r32 = T{3190:5902,20};
JuneAir_14_t_gshp_hex_out_r32 = T{3190:5902,21};
JuneAir_15_t_gshp_compsr_out= T{3190:5902,22};
JuneAir_16_t_gshp_compsr_in = T{3190:5902,23};
JuneAir_17_t_gshp_hex_in_water= T{3190:5902,24};
JuneAir_18_t_gshp_hex_out_water = T{3190:5902,25};
JuneAir_19_t_compos_compsr = T{3190:5902,26};
JuneAir_20_t_gshp_evap_out = T{3190:5902,27};
JuneAir_21_t_gshp_evap_in = T{3190:5902,28};
JuneAir_22_t_compos_hex_in_water = T{3190:5902,29};
JuneAir_44_t_gshp_rm = T{3190:5902,51};
JuneAir_45_t_gshp_amb = T{3190:5902,52};
JuneAir_46_t_gshp_wl = T{3190:5902,53};
JuneAir_47_t_gshp_surf = T{3190:5902,54};

%% Save


%% Plot 
figure(1)
plot(JuneAir_T_0,JuneAir_20_t_gshp_evap_out,'LineWidth', 2)
hold on
plot(JuneAir_T_0,JuneAir_21_t_gshp_evap_in,'LineWidth', 2)
plot(JuneAir_T_0,JuneAir_15_t_gshp_compsr_out,'LineWidth', 2)
plot(JuneAir_T_0,JuneAir_14_t_gshp_hex_out_r32,'LineWidth', 2)
title("Lap GSHP Temperature Data 20-1-2026 : degC")
xlim([0,max(JuneAir_T_0)+10])
ylabel('Temperature : degC')
xlabel('Time (s)')
legend('T Evap out : degC','T Evap in : degC',...
    'T Compsr out : degC','T Cond : degC')

figure(2)
plot(JuneAir_T_0,JuneAir_4_pow_gshp_gshp_ac,'LineWidth', 2)
hold on
plot(JuneAir_T_0,JuneAir_7_pow_gshp_gshp_cdu_fan,'LineWidth', 2)
title("Lap GSHP Power Data 20-1-2026 : kW")
legend('Power Fan Evap&Compsr : kW','Power Fan Cond : kW')
ylabel('Power : kW')
xlabel('Time (s)')
xlim([0,max(JuneAir_T_0)+10])








