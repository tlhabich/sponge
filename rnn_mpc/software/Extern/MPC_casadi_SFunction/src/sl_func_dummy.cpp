/*
Dummy code for defining the functions for the Simulink S function.
The content of the functions does not need to be known at the time of compilation.
The functions are integrated later via a program library.
*/

#ifdef DUMMY1   // robot system

    #include <sl_mpc.h>
    void SL_io_mpc_func(MPC_IN_type* inports, MPC_OUT_type* outports){}
    void SL_start_mpc_func(MPC_PARAM_type* params){}
    void SL_terminate_mpc_func(){}

#endif // DUMMY1
