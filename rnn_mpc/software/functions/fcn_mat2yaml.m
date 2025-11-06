function [MPC_params, NN_weights] = fcn_mat2yaml(model_folder, write_yaml, pred_horizon, t_sample_mpc_s, Q_pos, P_pos, R_p, dR_p, p_max_bar, p_min_bar, q_minmax_deg)
    % Write MPC parameters and neural network weights to YAML files for S-Function
    %
    %% Inputs:
    % write_yaml - flag indicating whether the YAML should be saved (true/false)
    % pred_horizon - discrete-time prediction horizon (number of time steps)
    % t_sample_mpc_s - sample time of the MPC block
    % Weights:
    % Q_pos - weighting matrix for states (position)
    % P_pos - weighting matrix for the final time step
    % R_p - weighting matrix for inputs (pressure)
    % dR_p - weighting matrix for pressure differences between time steps
    % p_max_bar - maximum pressure of the MPC block in bar
    % p_min_bar - minimum pressure of the MPC block in bar
    % q_minmax_deg - maximum angle in degrees
    %
    %% Outputs:
    % MPC_params - Contains all parameters for the MPC
    % NN_weights - Contains all parameters for the neural network

    %% path yaml toolbox
    addpath(genpath('Extern/YAMLMatlab_0.4.3'));
    addpath(genpath('functions'));

    dt_mpc = t_sample_mpc_s;
    
    % read in parameters
    current_path = string(pwd());
    pfad_GRU = current_path + '/models/'+model_folder;
    NN_weights = load(string(pfad_GRU) + '/GRU_weights.mat');
    NN_description = load(string(pfad_GRU) + '/GRU_params.mat');
    NN_weights = fcn_Transfer2Double(NN_weights);
    n_akt = NN_description.n_aktoren;

    % write parameter to yaml
    if write_yaml == 1
        WriteYaml(fullfile(current_path,'Extern/MPC_casadi_SFunction/include/models/sys_params.yaml'),NN_weights);
        disp('Model parameters saved!')
    else
        disp('Model parameters not saved')
    end

    % Initialize cost matrices
    Q = struct;
    for i = 1:n_akt
        variableName = ['Q_' num2str(i) '_' num2str(i)];
        Q.(variableName) = Q_pos;
    end

    P = struct;
    for i = 1:n_akt
        variableName = ['P_' num2str(i) '_' num2str(i)];
        P.(variableName) = P_pos;
    end

    R = struct;
    for i = 1:2*n_akt
        variableName = ['R_' num2str(i) '_' num2str(i)];
        R.(variableName) = R_p;
    end

    dR = struct;
    for i = 1:2*n_akt
        variableName = ['dR_' num2str(i) '_' num2str(i)];
        dR.(variableName) = dR_p;
    end

    %% MPC-Constraints
    p_min_Pa = p_min_bar * 1e5; % Maximum desired pressure in Pa
    p_max_Pa = p_max_bar * 1e5;
    q_minmax_rad = deg2rad(q_minmax_deg);% Maximum angle in rad
    xn_hilf_min = [repmat(-q_minmax_rad, 1, n_akt), repmat(p_min_Pa, 1, 2*n_akt)];
    xn_hilf_max = [repmat(q_minmax_rad, 1, n_akt), repmat(p_max_Pa, 1, 2*n_akt)];
    xs_hilf_min = fcn_MinMax(xn_hilf_min, NN_description.x_scaler_min, NN_description.x_scaler_max, [-1,1], 0);
    xs_hilf_max = fcn_MinMax(xn_hilf_max, NN_description.x_scaler_min, NN_description.x_scaler_max, [-1,1], 0);

    u_ub = struct; % Inputs Upper Bounds
    for i = 1:n_akt
        for j = 1:2
            variableName = ['p_ub_' num2str(i) '_' num2str(j)];
            u_ub.(variableName) = xs_hilf_max(i+j+n_akt); % p_max
        end
    end

    u_lb = struct; % Inputs Lower Bounds
    for i = 1:n_akt
        for j = 1:2
            variableName = ['p_lb_' num2str(i) '_' num2str(j)];
            u_lb.(variableName) = xs_hilf_min(i+j+n_akt); % p_min
        end
    end

    x_ub = struct; % States Upper Bounds
    m = 0;
    for i = 1:n_akt
        x_ub.(['pos_ub_' num2str(i)]) = xs_hilf_max(i);
    end

    x_lb = struct; % States Lower Bounds
    m = 0;
    for i = 1:n_akt
        x_lb.(['pos_lb_' num2str(i)]) = xs_hilf_min(i);
    end

    MPC_params = struct;
    MPC_params.dt_mpc = dt_mpc;
    MPC_params.pred_horizon = pred_horizon;
    MPC_params.Q = Q;
    MPC_params.dR = dR;
    MPC_params.R = R;
    MPC_params.P = P;
    MPC_params.x_ub = x_ub;
    MPC_params.x_lb = x_lb;
    MPC_params.u_ub = u_ub;
    MPC_params.u_lb = u_lb;

    % Write YAML files
    if write_yaml == 1
        WriteYaml(fullfile(current_path,'Extern/MPC_casadi_SFunction/include/models/mpc_params.yaml'),MPC_params);
        disp('MPC parameters saved!')
    else
        disp('MPC parameters not saved!')
    end

end % function