[folder, name, ext] = fileparts(which(mfilename));
builddir = fullfile(folder, 'build');
mkdir(builddir);
cd(builddir);

try
    def = legacy_code('initialize')
    def.SourceFiles = {'../ros_rt_interface/ros_rt_core/SL_func_dummy.cpp'};  
    def.HeaderFiles = {'SL_func.h'};
    def.IncPaths = {'../ros_rt_interface/ros_rt_core'};
    def.TargetLibFiles = {'libros_sl_interface.so'};
    def.LibPaths = {'../build'};

    def.SFunctionName = 'ros_rt_interface_pcu';
    
    def.StartFcnSpec = 'void SL_start_func()';
    def.OutputFcnSpec = 'void SL_io_func(SL_OUT_type u1[1], SL_IN_type y1[1])';
    def.TerminateFcnSpec = 'void SL_terminate_func()';

    legacy_code('sfcn_cmex_generate', def);
    legacy_code('compile', def, '-DDUMMY');   
    legacy_code('sfcn_tlc_generate', def);
    legacy_code('rtwmakecfg_generate', def);
    legacy_code('slblock_generate', def);
    
catch ME
    cd(folder);
    rethrow(ME)
end

cd(folder);
