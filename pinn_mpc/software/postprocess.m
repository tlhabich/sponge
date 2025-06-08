clc
clear 
close all
% dependency: simulink_signal2struct (https://github.com/SchapplM/matlab_toolbox)
run('../matlab_toolbox/matlab_tools_path_init.m');
resdir = fullfile(which(fileparts('postprocess.m')), 'results');
datastructpath = fullfile(resdir, 'measurements_struct.mat');
ExpDat = struct('t', [],"p_bar",[],"qddes_degs",[],"p_d_bar_MPC",[], 'q_deg_filt', [],'q_deg_raw', [], 'qd_degs', [], 'p_d_bar', [],'qdes_deg',[],'t_sol_s',[]);

matdatlist = dir(fullfile(resdir, 'measurement_data_*.mat'));
I = -1;
for i = 0:10000
  if exist(fullfile(resdir,sprintf('measurement_data_%d.mat', i)), 'file')
    I = i;
    break;
  end
end
if I == -1
  error('No data found');
end
for i = 1:length(matdatlist)
  dateiname_neu = sprintf('measurement_data_%d.mat', I+i-1);
  fprintf('Read file %d/%d: %s.\n', i,length(matdatlist), dateiname_neu);
  matdatpath = fullfile(resdir, dateiname_neu);
  tmp = load(matdatpath);
  sl_signal = simulink_signal2struct(tmp.ScopeData1);
  ExpDat = timestruct_append(ExpDat, sl_signal);
end
save(datastructpath, 'ExpDat', '-v7.3');

%% Plot

imesblau   = [0 80 155 ]/255; 
imesorange = [231 123 41 ]/255; 
imesgruen  = [200 211 23 ]/255;
for i=1:size(ExpDat.qdes_deg(:,1))
    if sum(ExpDat.qdes_deg(i,:))~=0
        first_desired = i;
        break;
    end
end
t_max=40/1e-2;
n_akt=5;

t_s=ExpDat.t(first_desired:first_desired+t_max)-ExpDat.t(first_desired);
t_sol_s=ExpDat.t_sol_s(first_desired:first_desired+t_max);
q_deg=ExpDat.q_deg_raw(first_desired:first_desired+t_max,:);
qd_degs=ExpDat.qd_degs(first_desired:first_desired+t_max,:);
qddes_degs=ExpDat.qddes_degs(first_desired:first_desired+t_max,:);
qdes_deg=ExpDat.qdes_deg(first_desired:first_desired+t_max,:);
p_d_bar_MPC=ExpDat.p_d_bar_MPC(first_desired:first_desired+t_max,:);
p_d_bar=ExpDat.p_d_bar(first_desired:first_desired+t_max,:);
p_bar=ExpDat.p_bar(first_desired:first_desired+t_max,:);
mae_deg=0;
figure("WindowState","maximized")
for i=1:n_akt
    subplot(3,2,i);
    plot(t_s, q_deg(:,i),'LineWidth',2,'Color',imesorange);
    hold on;
    grid on;
    plot(t_s, qdes_deg(:,i),'LineWidth',2,'Color',imesblau);
    temp_mae=mean(abs(qdes_deg(:,i)-q_deg(:,i)));
    title("MAE in deg: "+string(temp_mae));
    mae_deg=mae_deg+temp_mae/n_akt;
    ylabel("q"+string(i)+" in deg");
    xlabel("time in s");
    legend("meas.","des.","Location","northoutside","NumColumns",2);
end
mae_degs=0;
figure("WindowState","maximized")
for i=1:n_akt
    subplot(3,2,i);
    plot(t_s, qd_degs(:,i),'LineWidth',2,'Color',imesorange);
    hold on;
    grid on;
    plot(t_s, qddes_degs(:,i),'LineWidth',2,'Color',imesblau);
    temp_mae=mean(abs(qddes_degs(:,i)-qd_degs(:,i)));
    title("MAE in deg/s: "+string(temp_mae));
    mae_degs=mae_degs+temp_mae/n_akt;
    ylabel("qd"+string(i)+" in deg/s");
    xlabel("time in s");
    legend("meas.","des.","Location","northoutside","NumColumns",2);
end
figure("WindowState","maximized")
for i=1:n_akt
    subplot(3,2,i);    
    plot(t_s, p_bar(:,2*i-1),'LineWidth',2,'LineStyle',"--");
    hold on;
    plot(t_s, p_d_bar_MPC(:,2*i-1),'LineWidth',2);
    plot(t_s, p_bar(:,2*i),'LineWidth',2,'LineStyle',"--");
    plot(t_s, p_d_bar_MPC(:,2*i),'LineWidth',2);
    grid on;
    legend("p_{"+string(i)+"1}","p_{dMPC"+string(i)+"1}","p_{"+string(i)+"2}","p_{dMPC"+string(i)+"2}","Location","northoutside","NumColumns",4)
end
figure;
plot(t_s,t_sol_s);
title("mean solution time in s: "+string(mean(t_sol_s)));
disp("MAE q: "+string(mae_deg)+"deg");
disp("MAE qd: "+string(mae_degs)+"deg/s");
disp("mean solution freq: "+string(1/mean(t_sol_s))+"Hz");