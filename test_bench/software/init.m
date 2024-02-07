%open Simulink model and init all parameters
clear
clc
close all
%delete old measurements
delete('results/*');
%parameters
n_akt=3; %number of actuators
t_sample_s=1e-3; %global sample time
p_max_bar = 0.5; %pressure limit in each bellows
offset_q = [229.25, 150.1, 157.4, 0, 0];%encoder calibration
n_steps=7; %desired path
t_PWM_s=1/100;%for microvalves
p_supply_bar=0.3;%for modular system
%path
this_path = fileparts(which(mfilename));
cd(this_path);
addpath(fullfile(this_path, 'ros_rt_interface'));
addpath(fullfile(this_path, 'ros_rt_interface', 'build'));
this_tb_path = fileparts( mfilename('fullpath') );
addpath(this_tb_path);
addpath(genpath(fullfile(this_tb_path, "lib")));
run(fullfile(this_path, 'ros_rt_interface', 'pcu_ros_load_buses.m'));
%desired path for control experiment
dur=75;
slope=0.25;
for i_akt=1:n_akt
    rng(202308160+i_akt); %use date and actuator number as seed
    ramp_heights = round(-20+40*rand(n_steps,1),0);
    q_d_temp=[zeros(dur,1);[0:sign(ramp_heights(1))*slope:ramp_heights(1)]';ones(dur,1)*ramp_heights(1)];...
    for i=1:n_steps-1
        q_d_temp=[q_d_temp;[ramp_heights(i):sign(ramp_heights(i+1)-ramp_heights(i))*slope:ramp_heights(i+1)]';ones(dur,1)*ramp_heights(i+1)];
    end
    q_d_temp=[q_d_temp;[ramp_heights(n_steps):sign(-ramp_heights(n_steps))*slope:0]';zeros(dur,1)];
    if i_akt==1
        q1_d_deg_all=q_d_temp;
    elseif i_akt==2
        q2_d_deg_all=q_d_temp;
    elseif i_akt==3
        q3_d_deg_all=q_d_temp;
    elseif i_akt==4
        q4_d_deg_all=q_d_temp;
    else
        q5_d_deg_all=q_d_temp;
    end
end
open_system('./test_bench_sponge.mdl');
