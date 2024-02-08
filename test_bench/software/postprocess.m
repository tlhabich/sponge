% Postprocessing of the recorded data from external mode
% Reads in the consecutively numbered measurement data and saves them as a single mat file
% Dependency: simulink_signal2struct (https://github.com/SchapplM/matlab_toolbox)
clc
clear
resdir = fullfile(which(fileparts('postprocess.m')), 'results');
datastructpath = fullfile(resdir, 'measurements_struct.mat');
ExpDat = struct('t', [],"p_bar", [], 'q_deg', [], 'q_d_deg', [], 'p_d_bar', []);
matdatlist = dir(fullfile(resdir, 'measurement_data_*.mat'));
I = -1;
for i = 0:10000
  if exist(fullfile(resdir,sprintf('measurement_data_%d.mat', i)), 'file')
    I = i;
    break;
  end
end
if I == -1
  error('No measurement files with naming scheme found');
end
for i = 1:length(matdatlist)
  dateiname_neu = sprintf('measurement_data_%d.mat', I+i-1);
  fprintf('Read file %d/%d: %s.\n', i,length(matdatlist), dateiname_neu);
  matdatpath = fullfile(resdir, dateiname_neu);
  tmp = load(matdatpath);
  sl_signal = simulink_signal2struct(tmp.ScopeData1);
  ExpDat = timestruct_append(ExpDat, sl_signal);
end
save(datastructpath, 'ExpDat', '-v7.3');