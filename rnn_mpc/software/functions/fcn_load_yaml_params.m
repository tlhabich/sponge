addpath(genpath('Extern/YAMLMatlab_0.4.3'));

mpc_params = ReadYaml(mpc_param_config_file);
sys_params = ReadYaml(sys_param_config_file);

fnames = fieldnames(mpc_params);
n_params = length(fnames);
n_states = max(size(fieldnames(mpc_params.Q)));
n_controls = max(size(fieldnames(mpc_params.dR)));
n_outputs = max(size(fieldnames(mpc_params.dR)));
pred_horizon = mpc_params.pred_horizon;


% Create struct to write in xml
in_out_def = struct;

% create input ports for xml
in_out_def.matlab_in_out.IN.in1.name = 'state';
in_out_def.matlab_in_out.IN.in1.type = 'double';
in_out_def.matlab_in_out.IN.in1.dim = num2str(n_states);
in_out_def.matlab_in_out.IN.in2.name = 'ref';
in_out_def.matlab_in_out.IN.in2.type = 'double';
in_out_def.matlab_in_out.IN.in2.dim = [num2str(n_states) ' ' num2str(pred_horizon+1)];
in_out_def.matlab_in_out.IN.in4.name = 'on_off';
in_out_def.matlab_in_out.IN.in4.type = 'double';
in_out_def.matlab_in_out.IN.in4.dim = '1';
in_out_def.matlab_in_out.IN.in5.name = 'h_t_minus_1';
in_out_def.matlab_in_out.IN.in5.type = 'double';
in_out_def.matlab_in_out.IN.in5.dim = [num2str(hidden_dim) ' ' num2str(num_layer)];

% create output ports for xml
in_out_def.matlab_in_out.OUT.out1.name = 'mv';
in_out_def.matlab_in_out.OUT.out1.type = 'double';
in_out_def.matlab_in_out.OUT.out1.dim = num2str(n_controls);


for i = 1:n_params
    
    % read mpc params
    p_name = fnames{i};
    p_val = mpc_params.(fnames{i});
    if isstruct(p_val)
        dim = num2str(max(size(fieldnames(p_val))));
        ftmp = fieldnames(p_val);
        tmparr = [];
        for j = 1: str2num( dim)
            if (isempty(p_val.(ftmp{j})))
                if contains(ftmp{j}, 'ub')
                    tmparr = [tmparr; 1e3]; % TODO inf geht wegen codegen nicht
                elseif contains(ftmp{j}, 'lb')
                    tmparr = [tmparr; -1e3];
                else
                    tmparr = [tmparr; 0];
                end
            else
                tmparr = [tmparr; p_val.(ftmp{j})];
            end
        end
        p_val = tmparr;

    else
        dim = '1';
    end
    
   structParam.(p_name) = p_val;
   
    % create param ports for xml
    in_out_def.matlab_in_out.PARAM.(strcat('param',num2str(i))).name = fnames{i};
    in_out_def.matlab_in_out.PARAM.(strcat('param',num2str(i))).type = 'double';
    in_out_def.matlab_in_out.PARAM.(strcat('param',num2str(i))).dim = dim;
    
    
end

struct2xml(in_out_def,'Extern/MPC_casadi_SFunction/mFiles/build/sfunction_in_out.xml');
structParamBus = Simulink.Parameter(structParam);
structParamBus.DataType = 'Bus: MPC_PARAM_type';
