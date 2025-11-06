function [gru_weights] = fcn_Transfer2Double(gru_weights)
    % Convert GRU network weights to double precision
    %
    % Input:
    %   gru_weights - Structure containing GRU network parameters
    % Output:
    %   gru_weights - Same structure with all fields converted to double

    fields_to_convert = {'w_ir_l0', 'w_iz_l0', 'w_in_l0', 'b_ir_l0', 'b_iz_l0', 'b_in_l0', ...
                         'w_hr_l0', 'w_hz_l0', 'w_hn_l0', 'b_hr_l0', 'b_hz_l0', 'b_hn_l0', ...
                         'w_ir_l1', 'w_iz_l1', 'w_in_l1', 'b_ir_l1', 'b_iz_l1', 'b_in_l1', ...
                         'w_hr_l1', 'w_hz_l1', 'w_hn_l1', 'b_hr_l1', 'b_hz_l1', 'b_hn_l1', ...
                         'weight_linout', 'bias_linout'};

    for i = 1:numel(fields_to_convert)
        field_name = fields_to_convert{i};
        if isfield(gru_weights, field_name)
            gru_weights.(field_name) = double(gru_weights.(field_name));
        end
    end

end