#ifndef SL_FUNC_H
#define SL_FUNC_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
    double p_d_bar[10];
    double t_sol_s;
    
} SL_IN_type;
    
typedef struct{
    double ctrl_state;
    double q_deg[5];
    double q_des_deg[36];
    double qd_des_degs[36];
    double qd_degs[5];
} SL_OUT_type;

void SL_io_func(SL_OUT_type* sl_out, SL_IN_type* sl_in);
void SL_start_func();
void SL_terminate_func();

#ifdef __cplusplus
}
#endif

#endif
