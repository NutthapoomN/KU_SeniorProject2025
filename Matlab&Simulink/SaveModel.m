y_Actual = out.T_Room_Actual.Data ;
y_Sim = out.T_Room_Sim.Data;

T=out.T_Room_Sim.Time;
y_Actual=y_Actual;
y_Sim=y_Sim;
%(1:22)
plot(T, y_Sim, ...
    T, y_Actual,"o-")


mae_val = mean(abs(y_Actual - y_Sim));
mape_val = mean(abs((y_Actual - y_Sim) ./ y_Actual)) * 100;
rmse_val = sqrt(mean((y_Actual - y_Sim).^2));
ss_res = sum((y_Actual - y_Sim).^2);       
ss_tot = sum((y_Actual - mean(y_Actual)).^2);
r2_val = 1 - (ss_res / ss_tot);

fprintf('--- Evaluation Metrics ---\n');
fprintf('MAE:  %.4f\n', mae_val);
fprintf('MAPE: %.4f %%\n', mape_val);
fprintf('RMSE: %.4f\n', rmse_val);
fprintf('R2:   %.4f\n', r2_val);
fprintf('--------------------------\n');