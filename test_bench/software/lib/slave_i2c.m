% Config for Etherlab custom slave 
%
%
%

function rv = slave_i2c()

% Slave configuration

rv.SlaveConfig.vendor = 1946;                               
rv.SlaveConfig.product = hex2dec('00defede');               
rv.SlaveConfig.description = 'EasyCAT 32+32 rev 1';             
rv.SlaveConfig.sm = { ...                                   
    
    {0, 0, {
        {hex2dec('1600'), [
            hex2dec('0005'), hex2dec('01'),   8; ...
            hex2dec('0005'), hex2dec('02'),   8; ...
            hex2dec('0005'), hex2dec('03'),   8; ...
            hex2dec('0005'), hex2dec('04'),   8; ...
            hex2dec('0005'), hex2dec('05'),   8; ...
            hex2dec('0005'), hex2dec('06'),   8; ...
            hex2dec('0005'), hex2dec('07'),   8; ...
            hex2dec('0005'), hex2dec('08'),   8; ...
            hex2dec('0005'), hex2dec('09'),   8; ...
            hex2dec('0005'), hex2dec('0a'),   8; ...
            hex2dec('0005'), hex2dec('0b'),   8; ...
            hex2dec('0005'), hex2dec('0c'),   8; ...
            hex2dec('0005'), hex2dec('0d'),   8; ...
            hex2dec('0005'), hex2dec('0e'),   8; ...
            hex2dec('0005'), hex2dec('0f'),   8; ...
            hex2dec('0005'), hex2dec('10'),   8; ...
            hex2dec('0005'), hex2dec('11'),   8; ...
            hex2dec('0005'), hex2dec('12'),   8; ...
            hex2dec('0005'), hex2dec('13'),   8; ...
            hex2dec('0005'), hex2dec('14'),   8; ...
            hex2dec('0005'), hex2dec('15'),   8; ...
            hex2dec('0005'), hex2dec('16'),   8; ...
            hex2dec('0005'), hex2dec('17'),   8; ...
            hex2dec('0005'), hex2dec('18'),   8; ...
            hex2dec('0005'), hex2dec('19'),   8; ...
            hex2dec('0005'), hex2dec('1a'),   8; ...
            hex2dec('0005'), hex2dec('1b'),   8; ...
            hex2dec('0005'), hex2dec('1c'),   8; ...
            hex2dec('0005'), hex2dec('1d'),   8; ...
            hex2dec('0005'), hex2dec('1e'),   8; ...
            hex2dec('0005'), hex2dec('1f'),   8; ...
            hex2dec('0005'), hex2dec('20'),   8; ...
            ]}, ...
        }}, ...
    {1, 1, {
        {hex2dec('1a00'), [
            hex2dec('0006'), hex2dec('01'),   8; ...
            hex2dec('0006'), hex2dec('02'),   8; ...
            hex2dec('0006'), hex2dec('03'),   8; ...
            hex2dec('0006'), hex2dec('04'),   8; ...
            hex2dec('0006'), hex2dec('05'),   8; ...
            hex2dec('0006'), hex2dec('06'),   8; ...
            hex2dec('0006'), hex2dec('07'),   8; ...
            hex2dec('0006'), hex2dec('08'),   8; ...
            hex2dec('0006'), hex2dec('09'),   8; ...
            hex2dec('0006'), hex2dec('0a'),   8; ...
            hex2dec('0006'), hex2dec('0b'),   8; ...
            hex2dec('0006'), hex2dec('0c'),   8; ...
            hex2dec('0006'), hex2dec('0d'),   8; ...
            hex2dec('0006'), hex2dec('0e'),   8; ...
            hex2dec('0006'), hex2dec('0f'),   8; ...
            hex2dec('0006'), hex2dec('10'),   8; ...
            hex2dec('0006'), hex2dec('11'),   8; ...
            hex2dec('0006'), hex2dec('12'),   8; ...
            hex2dec('0006'), hex2dec('13'),   8; ...
            hex2dec('0006'), hex2dec('14'),   8; ...
            hex2dec('0006'), hex2dec('15'),   8; ...
            hex2dec('0006'), hex2dec('16'),   8; ...
            hex2dec('0006'), hex2dec('17'),   8; ...
            hex2dec('0006'), hex2dec('18'),   8; ...
            hex2dec('0006'), hex2dec('19'),   8; ...
            hex2dec('0006'), hex2dec('1a'),   8; ...
            hex2dec('0006'), hex2dec('1b'),   8; ...
            hex2dec('0006'), hex2dec('1c'),   8; ...
            hex2dec('0006'), hex2dec('1d'),   8; ...
            hex2dec('0006'), hex2dec('1e'),   8; ...
            hex2dec('0006'), hex2dec('1f'),   8; ...
            hex2dec('0006'), hex2dec('20'),   8; ...
            ]}, ...
        }}, ...
    };

% Port configuration

rv.PortConfig.input(1).pdo = [0, 0, 0, 0];
rv.PortConfig.input(1).pdo_data_type = 1008;

rv.PortConfig.input(2).pdo = [0, 0, 1, 0];
rv.PortConfig.input(2).pdo_data_type = 1008;

rv.PortConfig.input(3).pdo = [0, 0, 2, 0];
rv.PortConfig.input(3).pdo_data_type = 1008;

rv.PortConfig.input(4).pdo = [0, 0, 3, 0];
rv.PortConfig.input(4).pdo_data_type = 1008;

rv.PortConfig.input(5).pdo = [0, 0, 4, 0];
rv.PortConfig.input(5).pdo_data_type = 1008;

rv.PortConfig.input(6).pdo = [0, 0, 5, 0];
rv.PortConfig.input(6).pdo_data_type = 1008;

rv.PortConfig.input(7).pdo = [0, 0, 6, 0];
rv.PortConfig.input(7).pdo_data_type = 1008;

rv.PortConfig.input(8).pdo = [0, 0, 7, 0];
rv.PortConfig.input(8).pdo_data_type = 1008;

rv.PortConfig.input(9).pdo = [0, 0, 8, 0];
rv.PortConfig.input(9).pdo_data_type = 1008;

rv.PortConfig.input(10).pdo = [0, 0, 9, 0];
rv.PortConfig.input(10).pdo_data_type = 1008;

rv.PortConfig.input(11).pdo = [0, 0, 10, 0];
rv.PortConfig.input(11).pdo_data_type = 1008;

rv.PortConfig.input(12).pdo = [0, 0, 11, 0];
rv.PortConfig.input(12).pdo_data_type = 1008;

rv.PortConfig.input(13).pdo = [0, 0, 12, 0];
rv.PortConfig.input(13).pdo_data_type = 1008;

rv.PortConfig.input(14).pdo = [0, 0, 13, 0];
rv.PortConfig.input(14).pdo_data_type = 1008;

rv.PortConfig.input(15).pdo = [0, 0, 14, 0];
rv.PortConfig.input(15).pdo_data_type = 1008;

rv.PortConfig.input(16).pdo = [0, 0, 15, 0];
rv.PortConfig.input(16).pdo_data_type = 1008;

rv.PortConfig.input(17).pdo = [0, 0, 16, 0];
rv.PortConfig.input(17).pdo_data_type = 1008;

rv.PortConfig.input(18).pdo = [0, 0, 17, 0];
rv.PortConfig.input(18).pdo_data_type = 1008;

rv.PortConfig.input(19).pdo = [0, 0, 18, 0];
rv.PortConfig.input(19).pdo_data_type = 1008;

rv.PortConfig.input(20).pdo = [0, 0, 19, 0];
rv.PortConfig.input(20).pdo_data_type = 1008;

rv.PortConfig.input(21).pdo = [0, 0, 20, 0];
rv.PortConfig.input(21).pdo_data_type = 1008;

rv.PortConfig.input(22).pdo = [0, 0, 21, 0];
rv.PortConfig.input(22).pdo_data_type = 1008;

rv.PortConfig.input(23).pdo = [0, 0, 22, 0];
rv.PortConfig.input(23).pdo_data_type = 1008;

rv.PortConfig.input(24).pdo = [0, 0, 23, 0];
rv.PortConfig.input(24).pdo_data_type = 1008;

rv.PortConfig.input(25).pdo = [0, 0, 24, 0];
rv.PortConfig.input(25).pdo_data_type = 1008;

rv.PortConfig.input(26).pdo = [0, 0, 25, 0];
rv.PortConfig.input(26).pdo_data_type = 1008;

rv.PortConfig.input(27).pdo = [0, 0, 26, 0];
rv.PortConfig.input(27).pdo_data_type = 1008;

rv.PortConfig.input(28).pdo = [0, 0, 27, 0];
rv.PortConfig.input(28).pdo_data_type = 1008;

rv.PortConfig.input(29).pdo = [0, 0, 28, 0];
rv.PortConfig.input(29).pdo_data_type = 1008;

rv.PortConfig.input(30).pdo = [0, 0, 29, 0];
rv.PortConfig.input(30).pdo_data_type = 1008;

rv.PortConfig.input(31).pdo = [0, 0, 30, 0];
rv.PortConfig.input(31).pdo_data_type = 1008;

rv.PortConfig.input(32).pdo = [0, 0, 31, 0];
rv.PortConfig.input(32).pdo_data_type = 1008;

rv.PortConfig.output(1).pdo = [1, 0, 0, 0];
rv.PortConfig.output(1).pdo_data_type = 1008;

rv.PortConfig.output(2).pdo = [1, 0, 1, 0];
rv.PortConfig.output(2).pdo_data_type = 1008;

rv.PortConfig.output(3).pdo = [1, 0, 2, 0];
rv.PortConfig.output(3).pdo_data_type = 1008;

rv.PortConfig.output(4).pdo = [1, 0, 3, 0];
rv.PortConfig.output(4).pdo_data_type = 1008;

rv.PortConfig.output(5).pdo = [1, 0, 4, 0];
rv.PortConfig.output(5).pdo_data_type = 1008;

rv.PortConfig.output(6).pdo = [1, 0, 5, 0];
rv.PortConfig.output(6).pdo_data_type = 1008;

rv.PortConfig.output(7).pdo = [1, 0, 6, 0];
rv.PortConfig.output(7).pdo_data_type = 1008;

rv.PortConfig.output(8).pdo = [1, 0, 7, 0];
rv.PortConfig.output(8).pdo_data_type = 1008;

rv.PortConfig.output(9).pdo = [1, 0, 8, 0];
rv.PortConfig.output(9).pdo_data_type = 1008;

rv.PortConfig.output(10).pdo = [1, 0, 9, 0];
rv.PortConfig.output(10).pdo_data_type = 1008;

rv.PortConfig.output(11).pdo = [1, 0, 10, 0];
rv.PortConfig.output(11).pdo_data_type = 1008;

rv.PortConfig.output(12).pdo = [1, 0, 11, 0];
rv.PortConfig.output(12).pdo_data_type = 1008;

rv.PortConfig.output(13).pdo = [1, 0, 12, 0];
rv.PortConfig.output(13).pdo_data_type = 1008;

rv.PortConfig.output(14).pdo = [1, 0, 13, 0];
rv.PortConfig.output(14).pdo_data_type = 1008;

rv.PortConfig.output(15).pdo = [1, 0, 14, 0];
rv.PortConfig.output(15).pdo_data_type = 1008;

rv.PortConfig.output(16).pdo = [1, 0, 15, 0];
rv.PortConfig.output(16).pdo_data_type = 1008;

rv.PortConfig.output(17).pdo = [1, 0, 16, 0];
rv.PortConfig.output(17).pdo_data_type = 1008;

rv.PortConfig.output(18).pdo = [1, 0, 17, 0];
rv.PortConfig.output(18).pdo_data_type = 1008;

rv.PortConfig.output(19).pdo = [1, 0, 18, 0];
rv.PortConfig.output(19).pdo_data_type = 1008;

rv.PortConfig.output(20).pdo = [1, 0, 19, 0];
rv.PortConfig.output(20).pdo_data_type = 1008;

rv.PortConfig.output(21).pdo = [1, 0, 20, 0];
rv.PortConfig.output(21).pdo_data_type = 1008;

rv.PortConfig.output(22).pdo = [1, 0, 21, 0];
rv.PortConfig.output(22).pdo_data_type = 1008;

rv.PortConfig.output(23).pdo = [1, 0, 22, 0];
rv.PortConfig.output(23).pdo_data_type = 1008;

rv.PortConfig.output(24).pdo = [1, 0, 23, 0];
rv.PortConfig.output(24).pdo_data_type = 1008;

rv.PortConfig.output(25).pdo = [1, 0, 24, 0];
rv.PortConfig.output(25).pdo_data_type = 1008;

rv.PortConfig.output(26).pdo = [1, 0, 25, 0];
rv.PortConfig.output(26).pdo_data_type = 1008;

rv.PortConfig.output(27).pdo = [1, 0, 26, 0];
rv.PortConfig.output(27).pdo_data_type = 1008;

rv.PortConfig.output(28).pdo = [1, 0, 27, 0];
rv.PortConfig.output(28).pdo_data_type = 1008;

rv.PortConfig.output(29).pdo = [1, 0, 28, 0];
rv.PortConfig.output(29).pdo_data_type = 1008;

rv.PortConfig.output(30).pdo = [1, 0, 29, 0];
rv.PortConfig.output(30).pdo_data_type = 1008;

rv.PortConfig.output(31).pdo = [1, 0, 30, 0];
rv.PortConfig.output(31).pdo_data_type = 1008;

rv.PortConfig.output(32).pdo = [1, 0, 31, 0];
rv.PortConfig.output(32).pdo_data_type = 1008;

end
