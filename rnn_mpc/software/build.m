this_path = fileparts(which(mfilename));
cd(this_path);
DUMMY_TYPE = '-DDUMMY1'; 
run('Extern/MPC_casadi_SFunction/mFiles/generate_sl_block');   % creates the Simulink mask
addpath(genpath('Extern/MPC_casadi_SFunction/mFiles/build')) % adds the path of the created .mex function to Matlab
rtwbuild('test_bench_mpc')
