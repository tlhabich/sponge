function cellInfo = bus_SL_func_IN(header_file_name) 
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

IN = in_out_def.matlab_in_out.IN;
fieldnames_IN = fieldnames(IN);
n_IN = length(fieldnames_IN);

cellInfo_inputs = cell(n_IN,1);
% create cellarray with input and output definitions from .xml file
for i = 1 : n_IN
    cellInfo_inputs{i,1} = {IN.(fieldnames_IN{i}).name.Text, str2num(IN.(fieldnames_IN{i}).dim.Text), IN.(fieldnames_IN{i}).type.Text, -1, 'real', 'Sample', 'Fixed', [], [], '', ''};
end

cellInfo = { ... 
  { ... 
    'MPC_IN_type', ... 
    header_file_name, ... 
    '', ... 
    'Auto', ... 
    '-1', {... 
    cellInfo_inputs{:,1};
    } ...
  } ...
}'; 

% if ~suppressObject 
    % Create bus objects in the MATLAB base workspace 
    Simulink.Bus.cellToObject(cellInfo) 
% end 
