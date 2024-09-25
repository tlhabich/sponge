#ifndef SL_MPC_H
#define SL_MPC_H

#define N_STATES 5
#define N_INPUTS 10
#define PRED_HORIZON 4

#define HIDDEN_STATES 59
#define HIDDEN_LAYERS 1

#ifdef __cplusplus
extern "C"{
#endif

typedef struct{
    double state[N_STATES]; // current measured state
    double ref[N_STATES][PRED_HORIZON+1]; // desired state vector (positions)
    double on_off;  // to skipp calculations in 'SL_io_mpc_func'
    double h_t_minus_1[HIDDEN_STATES][HIDDEN_LAYERS]; // current hidden states of gru net
} MPC_IN_type;
    
typedef struct{
    double mv[N_INPUTS];                      // computed control input
} MPC_OUT_type;

typedef struct {
    double dt_mpc;             // sample time of the MPC controller
    double pred_horizon;       // prediction horizon in x time steps
    double Q[N_STATES];        // state stage Cost weighting factor
    double dR[N_INPUTS];       // input difference stage Cost weighting factor
    double R[N_INPUTS];        // input stage Cost weighting factor
    double P[N_STATES];        // Terminal Cost weighting factor
    double x_ub[N_STATES];     // upper bound of state constraint
    double x_lb[N_STATES];     // lower bound of state constraint
    double u_ub[N_INPUTS];     // upper bound of input constraint
    double u_lb[N_INPUTS];     // lower bound of input constraint
} MPC_PARAM_type;

void SL_start_mpc_func(MPC_PARAM_type* params);
void SL_io_mpc_func(MPC_IN_type* inports, MPC_OUT_type* outports);
void SL_terminate_mpc_func();

#ifdef __cplusplus
}
#endif

#endif // SL_MPC_H
