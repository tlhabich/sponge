function [x_k_plus_1, h_k] = fcn_GRU_n_layer(x_k, h_k_minus_1, gru_weights)
% Function computes the forward pass of a GRU network with n GRU layers and
% a linear output layer.
%
% Inputs:
% x_k - States and inputs at time step k, must have dimensions [input_dim, 1].
%       Example: q1, q2, ..., p11d, p12d, p21d, p22d, ...
% h_k_minus_1 - Hidden states from the previous time step, must have dimensions [hidden_dim, num_layer].
% gru_weights - Parameters of the GRU network

    num_layer = (numel(fieldnames(gru_weights))-2)/12; % Anzahl an weights - 2(linear output layer) / 12 (Anzahl an weights pro GRU Layer)
    hidden_dim = size(gru_weights.weight_linout,2);
    w = gru_weights; % simpler notation
    h_k = zeros(hidden_dim, num_layer); % initialize hidden states

    for layer = 0:(num_layer-1) % loop through layer
        h_k_minus_1_layer = h_k_minus_1(:, layer+1);

        % Reset Gate, Update Gate und New Gate
        r_k = sigmoid(w.(['w_ir_l' num2str(layer)]) * x_k + w.(['b_ir_l' num2str(layer)])' + ...
                       w.(['w_hr_l' num2str(layer)]) * h_k_minus_1_layer + w.(['b_hr_l' num2str(layer)])');
        z_k = sigmoid(w.(['w_iz_l' num2str(layer)]) * x_k + w.(['b_iz_l' num2str(layer)])' + ...
                       w.(['w_hz_l' num2str(layer)]) * h_k_minus_1_layer + w.(['b_hz_l' num2str(layer)])');
        n_k = tanh(w.(['w_in_l' num2str(layer)]) * x_k + w.(['b_in_l' num2str(layer)])' + ...
                   r_k .* (w.(['w_hn_l' num2str(layer)]) * h_k_minus_1_layer + w.(['b_hn_l' num2str(layer)])'));
        % update state and hidden state
        h_k(:, layer+1) = (1 - z_k) .* n_k + z_k .* h_k_minus_1_layer;
        x_k = h_k(:, layer+1);
    end
    % linear output layer
    x_k_plus_1 = w.weight_linout * h_k(:, end) + w.bias_linout';
end

function out = sigmoid(x)
    % applies the sigmoid function to the input vector x
    out = 1 ./ (1 + exp(-x));
end