function cellInfo = bus_SL_func_PARAM(header_file_name) 
% BUS_SL_OUT returns a cell array containing bus object information 
% 
% Optional Input: 'false' will suppress a call to Simulink.Bus.cellToObject 
%                 when the MATLAB file is executed. 
% The order of bus element attributes is as follows:
%   ElementName, Dimensions, DataType, SampleTime, Complexity, SamplingMode, DimensionsMode, Min, Max, DocUnits, Description 

% suppressObject = false; 
% if nargin == 1 && islogical(varargin{1}) && varargin{1} == false 
%     suppressObject = true; 
% elseif nargin > 1 
%     error('Invalid input argument(s) encountered'); 
% end 

% add functions to read .xml
this_path = fileparts(which(mfilename));
addpath(fullfile(this_path, 'lib'));
in_out_def = xml2struct("build/sfunction_in_out.xml");   %read xml to workspace

PARAM = in_out_def.matlab_in_out.PARAM;
fieldnames_PARAM = fieldnames(PARAM);
n_PARAM = length(fieldnames_PARAM);

cellInfo_params = cell(n_PARAM,1);
% create cellarray with input and output definitions from .xml file
for i = 1 : n_PARAM
    cellInfo_params{i,1} = {PARAM.(fieldnames_PARAM{i}).name.Text, str2num(PARAM.(fieldnames_PARAM{i}).dim.Text), PARAM.(fieldnames_PARAM{i}).type.Text, -1, 'real', 'Sample', 'Fixed', [], [], '', ''};
end

cellInfo = { ... 
  { ... 
    'MPC_PARAM_type', ... 
    header_file_name, ... 
    '', ... 
    'Auto', ... 
    '-1', {... 
    cellInfo_params{:,1};
    } ...
  } ...
}'; 

% if ~suppressObject 
    % Create bus objects in the MATLAB base workspace 
    Simulink.Bus.cellToObject(cellInfo) 
% end 
