
[folder, name, ext] = fileparts(which(mfilename));
builddir = fullfile(folder, 'build'); % build folder in the same directory as this script
mkdir(builddir);
cd(builddir);

% Get target library name with .so ending
libName = dir(fullfile('', '*.so')); % Under MacOS .dylib; Under Ubuntu .so
TargetLibName = libName.name;

% Create name of Sfunction block by cutty TargetLibName. Use Indexing since
% libName is always in the same format
BlockName = TargetLibName(4:end-3); % for .dylib change to -6, for .so to -3

% Generate Simulink block from CPP files with precompiled TargetLib 
try
    def = legacy_code('initialize');
    def.SourceFiles = {'../../src/sl_func_dummy.cpp'};   % The specification of this file is necessary despite the shared library, as it is only compatible with ARM.
    def.HeaderFiles = {header_file_name};
    def.IncPaths = {'../../include/sl_function/', casadi_include_path};
    def.TargetLibFiles = {TargetLibName, 'libcasadi.so'}; % Under MacOS .dylib; Under Ubuntu .so
    def.LibPaths = {'../build', casadi_lib_path};
    

    def.SFunctionName = BlockName;
    
    def.StartFcnSpec = 'void SL_start_mpc_func(MPC_PARAM_type p1[1])';
    def.OutputFcnSpec = 'void SL_io_mpc_func(MPC_IN_type u1[1], MPC_OUT_type y1[1])';
    def.TerminateFcnSpec = 'void SL_terminate_mpc_func()';
    
    def.SampleTime = structParam.dt_mpc;      
%     def.SampleTime = 'parameterized';   

    legacy_code('sfcn_cmex_generate', def);
    legacy_code('compile', def, DUMMY_TYPE);    % Erstellt MEX-File. Ist leider nötig
    legacy_code('sfcn_tlc_generate', def);
    legacy_code('rtwmakecfg_generate', def);
    legacy_code('slblock_generate', def);
%     pause(.1);
%     bdclose;    % Close simulink model containing mask
catch ME
    cd(folder)
    throw(ME)
end

cd(folder)
close_system('untitled',0)
