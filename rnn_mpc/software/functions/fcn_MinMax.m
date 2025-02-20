function data_scaled = fcn_MinMax(data, data_min, data_max, feature_range, inverse)   
    % Apply min-max scaling to data
    %
    % Inputs:
    %   data - Data to scale
    %   data_min - Minimum values for each feature
    %   data_max - Maximum values for each feature  
    %   feature_range - Target range [min max] for scaled data
    %   inverse - If true, apply inverse scaling
    %
    % Output:
    %   data_scaled - Scaled data

    min = single(feature_range(1)); % -1
    max = single(feature_range(2)); % 1
    features = size(data,2);
    data_std = zeros(size(data));
    data_scaled = zeros(size(data));
    
    if inverse == 0 % min-max transformation
        for i = 1:features
            data_std(:,i) = (data(:,i) - data_min(:,i)) ./ (data_max(:,i) - data_min(:,i));
            data_scaled(:,i) = data_std(:,i) * (max - min) + min;
        end
    else % inverse min-max transformation
        for i = 1:features
            data_scaled(:,i) = data(:,i);
            data_std(:,i) = (data_scaled(:,i) - min) / (max - min);
            data(:,i) = data_std(:,i) .* (data_max(:,i) - data_min(:,i)) + data_min(:,i);
            data_scaled(:,i) = data(:,i);
        end
    end

    data_scaled = double(data_scaled);
end
