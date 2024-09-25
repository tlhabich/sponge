function trajectories = fcn_generateRampTrajectories(n_akt, t_sample_s, amplitude_deg, t_ramp, t_const, steps)
    % Generates ramp trajectories for multiple actuators
    %
    % Inputs:
    % n_akt - number of actuators
    % t_sample_s - sample time in seconds
    % amplitude_deg - amplitude of the ramp in degrees
    % t_ramp - ramp time in seconds
    % t_const - constant signal time in seconds
    % steps - number of steps in the ramp
    %
    % Outputs:
    % trajectories - matrix of generated ramp trajectories for each actuator
    
    amplitude = deg2rad(amplitude_deg);
    t_ges = t_ramp+t_const;
    t_sample_ramp = t_ramp / t_sample_s;
    t_sample_const = t_const / t_sample_s;
    steps_position = linspace(-amplitude, amplitude, steps);
    trajectories = zeros((steps+1)*(t_ges/t_sample_s), n_akt);

    for idx = 1:n_akt
        x_traj = [];
        steps_rand = steps_position(randperm(size(steps_position,2)));
        x_ramp_prev = 0;
        steps_rand = [steps_rand, 0];
        for i = 1:length(steps_rand)
            ramp = linspace(x_ramp_prev,steps_rand(i),t_sample_ramp);
            const = ones(1,t_sample_const)*steps_rand(i);
            x_traj = horzcat(x_traj,ramp, const);
            x_ramp_prev = steps_rand(i);
        end

        trajectories(:, idx) = x_traj';
    end
    trajectories = double(trajectories);
end
