function cellInfo = bus_SL_func_OUT(header_file_name) 
% BUS_SL_IN returns a cell array containing bus object information 
% 
% Optional Input: 'false' will suppress a call to Simulink.Bus.cellToObject 
%                 when the MATLAB file is executed. 
% The order of bus element attributes is as follows:
%   ElementName, Dimensions, DataType, SampleTime, Complexity, SamplingMode, DimensionsMode, Min, Max, DocUnits, Description 
% 
% suppressObject = false; 
% if nargin == 1 && islogical(varargin{1}) && varargin{1} == false 
%     suppressObject = true; 
% elseif nargin > 1 
%     error('Invalid input argument(s) encountered'); 
% end 

% add functions to read .xml
this_path = fileparts(which(mfilename));
addpath(fullfile(this_path, 'lib'));
in_out_def = xml2struct("build/sfunction_in_out.xml");  %read xml to workspace

OUT = in_out_def.matlab_in_out.OUT;
fieldnames_OUT = fieldnames(OUT);
n_OUT = length(fieldnames_OUT);

cellInfo_outputs = cell(n_OUT,1);

% create cellarray with input and output definitions from .xml file
for i = 1 : n_OUT
    cellInfo_outputs{i,1} = {OUT.(fieldnames_OUT{i}).name.Text, str2num(OUT.(fieldnames_OUT{i}).dim.Text), OUT.(fieldnames_OUT{i}).type.Text, -1, 'real', 'Sample', 'Fixed', [], [], '', ''};
end

cellInfo = { ... 
  { ... 
    'MPC_OUT_type', ... 
    header_file_name, ... 
    '', ... 
    'Auto', ... 
    '-1', {... 
cellInfo_outputs{:,1};
} ...
  } ...
}'; 

% if ~suppressObject 
    % Create bus objects in the MATLAB base workspace 
    Simulink.Bus.cellToObject(cellInfo) 
% end 
