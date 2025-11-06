% open Simulink model and init all parameters
clear
clc
close all
%delete old measurements
delete('results/*');
%parameters
n_akt=5; %number of actuators
f_sample_Hz = 20; % global sample frequency
f_sample_mpc_Hz = 5; % mpc sample frequency
t_sample_mpc_s = 1/f_sample_mpc_Hz ;
t_sample_s= 1/f_sample_Hz;
N = 4; % prediction horizon for mpc
Q_pos = 6; % stage Cost
P_pos = 5; % terminal Cost
R_p = 0.002; % inputs Cost (pressure)
dR_p = 1.6; % weighting matrix for pressure differences between time steps
p_min_bar = 0.7; % pressure limit in each bellows
p_max_bar = 0.7; % pressure limit in each bellows
q_minmax_deg = 18; % maximum angle in degrees (negative is minimum angle)
write_yaml = true; % flag whether yaml files should be saved
offset_q = [136,221.2,148.39,227.55,81.38]; % encoder calibration
moving_average = 3; % moving average filter for Input

%path
this_path = fileparts(which(mfilename));
cd(this_path);
addpath(fullfile(this_path, 'ros_rt_interface'));
addpath(fullfile(this_path, 'ros_rt_interface', 'build'));
this_tb_path = fileparts( mfilename('fullpath') );
addpath(this_tb_path);
addpath(genpath(fullfile(this_tb_path, "lib")));
run(fullfile(this_path, 'ros_rt_interface', 'pcu_ros_load_buses.m'));
addpath(genpath('Extern'));
addpath(genpath('model'))
addpath('functions')

% casadi
casadi_lib_path = '/usr/local/lib';
casadi_include_path = '/usr/local/lib/cmake/casadi';
sys_param_config_file = fullfile(this_path, 'Extern/MPC_casadi_SFunction/include/models/sys_params.yaml');
header_file_name = 'sl_mpc.h';
mpc_param_config_file = fullfile(this_path, 'Extern/MPC_casadi_SFunction/include/models/mpc_params.yaml');
if ~isfolder( casadi_lib_path)
    error('no valid casadi_lib_path file path')
end
if ~isfolder( casadi_include_path)
    error('no valid casadi_include_path file path')
end
if ~isfile( sys_param_config_file)
    error('no valid sys param file path')
end
if ~isfile( mpc_param_config_file)
    error('no valid mpc_param_config_file path')
end

% read in learned parameters
gru_params = load('models/GRU_params.mat');
hidden_dim = double(gru_params.hidden_dim);
num_layer = double(gru_params.num_layer);
xdim = double(gru_params.xdim);
x_scaler_max = double(gru_params.x_scaler_max); % scaled states + inputs
x_scaler_min = double(gru_params.x_scaler_min); 
y_scaler_max = double(gru_params.y_scaler_max); % scaled states
y_scaler_min = double(gru_params.y_scaler_min);
u_scaler_max = double(gru_params.x_scaler_max(:,xdim+1:end)); % scaled inputs
u_scaler_min = double(gru_params.x_scaler_min(:,xdim+1:end));

% write parameters in yaml files
[mpc_params, gru_weights] = fcn_mat2yaml(model_folder, write_yaml, N, t_sample_mpc_s, Q_pos, P_pos, R_p, dR_p, p_max_bar, p_min_bar, q_minmax_deg);
fcn_load_yaml_params;

% bus definitions for mpc block
addpath(fullfile(this_path, 'Extern/MPC_casadi_SFunction/mFiles'));
bus_SL_func_IN(header_file_name);
bus_SL_func_OUT(header_file_name);
bus_SL_func_PARAM(header_file_name);

% desired path for control experiment
amplitude_ramp_deg = 13; % maximum position in degrees
steps = 39; % number of different positions
t_ramp_s = 3; % time for ramp up
t_const_s = 1; % time for constant position
rng(41); % random seed
xn_goal_ramp = fcn_generateRampTrajectories(n_akt, t_sample_mpc_s, amplitude_ramp_deg, t_ramp_s, t_const_s, steps);
zeros_arr = zeros(60, 5);
xn_goal_ges = [xn_goal_ramp; zeros_arr];
xs_goal_ges = fcn_MinMax(xn_goal_ges, y_scaler_min, y_scaler_max, [-1,1], false);

% initial values for mpc
h_k_init = zeros(hidden_dim, num_layer); % Hidden States mit 0 initialisieren
xn_init = zeros(1,n_akt);
xn_goal_i = zeros(N,n_akt);
xs_goal_i = zeros(N,n_akt);

open_system('./test_bech_mpc.mdl')
