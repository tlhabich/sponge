clear
clc
close all
delete('results/*');

%% input
n_akt=5; % number of actuators
t_sample_s=1e-3; % sample time
f_tp_Hz = 1; % low-pass freq
p_max_bar = 0.7; % max pressure
offset_q_offline = [266.1,185.3,200.5,240.7,168.1]; % encoder calibration
pi_flag=0; % pi control
P_gain=[4,6,11,11,10]*1e-2*1;
I_gain=[4,4,8,6,11]*1e-2*1;
mpc_flag=1; % MPC
% desired trajectory
qmax_deg=18;
n_ramps=45;
t_sample_MPC_s=1/50;
ramp_const_s=0.0;
ramp_dur_max_s=1.6;
ramp_dur_min_s=0.4;
k_movmean=15; % smooth trajecotry
pred_horizon=7; % only for data transfer (prediction horizon of MPC is set in ROS)
%% main
this_path = fileparts(which(mfilename));
cd(this_path);
addpath(fullfile(this_path, 'ros_rt_interface'));
addpath(fullfile(this_path, 'ros_rt_interface', 'build'));
this_tb_path = fileparts( mfilename('fullpath') );
addpath(this_tb_path);
addpath(genpath(fullfile(this_tb_path, "lib")));
run(fullfile(this_path, 'ros_rt_interface', 'pcu_ros_load_buses.m'));

Q=nan(ramp_const_s/t_sample_MPC_s+n_ramps*(ramp_dur_max_s/t_sample_MPC_s+ramp_const_s/t_sample_MPC_s),n_akt);
Qd=nan(ramp_const_s/t_sample_MPC_s+n_ramps*(ramp_dur_max_s/t_sample_MPC_s+ramp_const_s/t_sample_MPC_s-1),n_akt);
q_lenghts=zeros(n_akt,1);
for i_akt=1:n_akt
    rng(300+i_akt); % seed for trajectory
    alternate_sign=(-1).^(1:n_ramps);
    rand_deg=qmax_deg*rand(n_ramps,1);
    ramp_heights_deg = rand_deg.*alternate_sign';
    ramp_dur_ramps=round((ramp_dur_min_s+(ramp_dur_max_s-ramp_dur_min_s)*rand(n_ramps,1))/t_sample_MPC_s,0);
    qdesi_deg=zeros(2/t_sample_MPC_s,1);%q=0 for 2s at beginning
    for i_ramp=1:n_ramps
        qdesi_deg=[qdesi_deg;...
        (qdesi_deg(end,1)+(ramp_heights_deg(i_ramp)-qdesi_deg(end,1))*linspace(0,1,ramp_dur_ramps(i_ramp)))';...
        ones(ramp_const_s/t_sample_MPC_s,1)*ramp_heights_deg(i_ramp)];
    end
    q_lenghts(i_akt)=length(qdesi_deg);
    if(k_movmean>0)
        qdesi_deg=movmean(qdesi_deg,k_movmean);
    end
    qddesi_degs= diff(qdesi_deg)/t_sample_MPC_s;
    Q(1:length(qdesi_deg),i_akt)=qdesi_deg;
    Qd(1:length(qdesi_deg)-1,i_akt)=qddesi_degs;  
end
Q=Q(1:min(q_lenghts),:);
Qd=Qd(1:min(q_lenghts)-1,:);
time_s=t_sample_MPC_s*(0:length(Q)-1);
% plot desired trajectory
figure;
for i_akt=1:n_akt
    subplot(round(n_akt/2),2,i_akt);
    plot(time_s,Q(:,i_akt));
    hold on;
    plot(time_s(1:end-1),Qd(:,i_akt));
    legend("q_"+i_akt,"qd_"+i_akt);
    grid on;
end
%split for prediction horizon
Q_total=nan(pred_horizon,n_akt,length(Q)-pred_horizon);
Qd_total=nan(pred_horizon,n_akt,length(Qd)-pred_horizon);
for i=1:length(Q)-pred_horizon
    Q_total(:,:,i)=Q(i:i+pred_horizon-1,:);
end
for i=1:length(Qd)-pred_horizon
    Qd_total(:,:,i)=Qd(i:i+pred_horizon-1,:);
end
open_system('./test_bench.mdl')
