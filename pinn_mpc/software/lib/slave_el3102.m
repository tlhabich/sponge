function rv = slave_el3102()

% Slave configuration

rv.SlaveConfig.vendor = 2;
rv.SlaveConfig.product = hex2dec('0c1e3052');
rv.SlaveConfig.description = 'EL3102';
rv.SlaveConfig.sm = { ...
    {0, 0, {
        }}, ...
    {1, 1, {
        }}, ...
    {2, 0, {
        }}, ...
    {3, 1, {
        {hex2dec('1a00'), [
            hex2dec('3101'), hex2dec('01'),   8; ...
            hex2dec('3101'), hex2dec('02'),  16; ...
            ]}, ...
        {hex2dec('1a01'), [
            hex2dec('3102'), hex2dec('01'),   8; ...
            hex2dec('3102'), hex2dec('02'),  16; ...
            ]}, ...
        }}, ...
    };

% Port configuration

rv.PortConfig.output(1).pdo = [3, 0, 0, 0];
rv.PortConfig.output(1).pdo_data_type = 1008;

rv.PortConfig.output(2).pdo = [3, 0, 1, 0];
rv.PortConfig.output(2).pdo_data_type = 1016;

rv.PortConfig.output(3).pdo = [3, 1, 0, 0];
rv.PortConfig.output(3).pdo_data_type = 1008;

rv.PortConfig.output(4).pdo = [3, 1, 1, 0];
rv.PortConfig.output(4).pdo_data_type = 1016;

end
