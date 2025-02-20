#ifndef SPONGE_MPC_H
#define SPONGE_MPC_H

#include <iostream>
#include <models/ModelBase.h>

namespace sponge
{
class SPONGE_MPC
{
private:
    bool first_run;

    ModelBase *model_;      // System to control. Can be RobotSystem
                            //    or any other class whick inherits from ModelBase
    
    
    MX costFunction_;   // Symbolic definition of the cost functioni
    MX optimVars_;      // all optimization variables: X_ (nxN) and U_ (mxN) stacked together
    MX constraints_;    // Symbolic general nonlinear constraints, used for continuity conditions of MS
    MX P_;              // Symbolic vector for measured x0 and reference x_ref


    Opti opti_;             // Create Opti object
    string solver_name_;    // Name of the solver (here ipopt)

    Dict options, coptions;
    // Dict solverOpts_;      // Options for Ipopt (see: https://coin-or.github.io/Ipopt/OPTIONS.html)
    //Dict nlpOpts_;          // Options for the solver function nlpSolver_
    
    
    MX X_, U_;              // Optimization variabes (state and input sequence over prediction horizon)
    MX X_K_, U_K_;          // Initial guess for the optimization variables

    MX H_K_;                // Hidden States of GRU net
    MX X_GOAL_;             // reference Value

    MX x_nexk;
    MX h_k;


    int N_;                 // Prediction Horizon length (in samples)
    double dt_;             // Sample time for MPC

    MX ubx_, lbx_;          // Numeric values of box constraints on optimization varaibles (for state and input constraints)
    MX lbg_, ubg_;          // Numeric values of general Nonlinear Constraints (continuity condition)

    DM solution_;           // Solution vector of the optimization problem
    DM x_opt;               // Optimierte States
    DM u_opt;               // Optimierte Inputs (mv)
    DM X0_, U0_; 

    int n_states_, n_controls_, n_outputs_;  // nr of states, inputs and outputs
    int n_hidden_states_, n_hidden_layers_;  // nr of hidden states and hidden layers
    
    Slice all;  // to access a whole row or colon in a casadi matrix SX, MX or DM

    void CreateNLPSolver(double max_computation_time);  // creates the NLP solver (automatically called in constructor)

public:
    SPONGE_MPC(ModelBase *model,
        double pred_horizon,
        double dt,
        double max_computation_time);

    ~SPONGE_MPC();
    
    // Functions to add Costterms (optional)
    void addStateStageCost(DM Q);  // adds a quadratic stage cost term of the form x^TQx, where x = (x(k)-xs)
    void addInputStageCost(DM R);   // adds a quadratic stage cost term of the form u^TRu, where  u = (u(k)-us)
    void addTerminalCost(DM P);   // adds a quadratic terminal cost term of the form x^TPx where x = (x(T) - xs(T))
    void addStateDifferenceCost(DM dQ);
    void addInputDifferenceCost(DM dR);
    void addMeanInputCost(DM R_p_mean, DM p_mean_scale);

    // Functions to add constraints (optional)    
    void addStateConstraints(DM x_ub, DM x_lb); // to add box state constraints with x_ub/x_lb nx1
    void addInputConstraints(DM u_ub, DM u_lb); // to add box input constraints with x_ub/x_lb mx1

    DM ComputeControlInput(DM x_meas, DM x_ref, DM h_k_minus_1);    // computes the next control input by solving the optimization problem
};

} // namespace sponge

#endif // SPONGE_MPC_H
