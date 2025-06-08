[folder, ~, ~] = fileparts(which(mfilename));
filedir = fullfile(folder, 'ros_rt_core');

run(fullfile(filedir, 'bus_SL_IN.m'))
run(fullfile(filedir, 'bus_SL_OUT.m'))
