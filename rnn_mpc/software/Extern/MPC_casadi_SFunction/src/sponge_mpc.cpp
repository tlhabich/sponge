#include "mpc_lib/sponge_mpc.h"

namespace sponge
{
  using namespace std;
  using namespace casadi;
  
SPONGE_MPC::SPONGE_MPC(ModelBase *model, double pred_horizon, double dt, double max_computation_time)
{

    model_ = model;
    n_states_ = model_->get_n_states();
    n_controls_ = model_->get_n_controls();
    n_outputs_ = model_->get_n_outputs();

    n_hidden_states_ = model_->get_n_hidden_states();
    n_hidden_layers_ = model_->get_n_hidden_layers();
    
    dt_ = dt;
    N_ = pred_horizon;
    solver_name_ = "ipopt";
    CreateNLPSolver(max_computation_time);
    first_run = true;
    
}


SPONGE_MPC::~SPONGE_MPC()
{
}

void SPONGE_MPC::CreateNLPSolver(double max_computation_time)
{   
    std::cout << "Creating NLP solver with Opti stack" << std::endl;
    Slice all;

    opti_ = Opti();
    
    options["print_level"] = 0;
    coptions["print_time"] = false;
    opti_.solver("ipopt", coptions, options);
    
    // Decision variables
    X_ = opti_.variable(n_states_, N_+1);    // State trajectory
    U_ = opti_.variable(n_controls_, N_);    // Control trajectory

    // Parameters
    X_GOAL_ = opti_.parameter(n_states_, N_+1);      // Reference trajectory
    H_K_ = opti_.parameter(n_hidden_states_, n_hidden_layers_);  // Hidden states
    X_K_ = opti_.parameter(n_states_, 1);     // Current state
    U_K_ = opti_.parameter(n_controls_, 1);   // Current input

    // Initial constraints
    opti_.subject_to(X_(all,0) == X_K_);    // First optimized state must match measured state

    // Cost function
    costFunction_ = 0;
    opti_.minimize(costFunction_);

    // Dynamic constraints
    for (int k = 0; k < N_; ++k) {
        x_nexk = model_->eval(X_(all,k), U_(all,k), H_K_);
        opti_.subject_to(X_(all, k + 1) == x_nexk);
    }
    std::cout << "Opti stack configuration: " << opti_ << std::endl;
}

    void SPONGE_MPC::addStateConstraints(DM x_ub, DM x_lb)
    {   
        // Function to add box constraints on predicted state
        for (int i = 0; i < n_states_; i++)
        {
            opti_.subject_to(x_lb(i) <= X_(i,all) <= x_ub(i));
        }

    }

    void SPONGE_MPC::addInputConstraints(DM u_ub, DM u_lb)
    {   
        // Function to add box constraints on predicted input        
        for (int i = 0; i < n_controls_; i++)
        {
            opti_.subject_to(u_lb(i) <= U_(i,all) <= u_ub(i));
        }
    }

    void SPONGE_MPC::addStateStageCost(DM Q)
    {   
        // create symbolic variables stage cost at t=k
        MX x_cost = MX::sym("x_cost", n_states_);

        for (int k = 0; k < N_; k++)
        {   
            // Create Stage Cost term for difference to reference state trajectory (x(k)-x_ref(k)) Q (x(k)-x_ref(k))
            x_cost = X_(all,k) - X_GOAL_(all,k);
            
            costFunction_ += mtimes(mtimes(x_cost.T(),Q),x_cost);
        }

        // update the NLP
        opti_.minimize(costFunction_);
        std::cout<< "added State Stage Cost" << std::endl;
    }


    void SPONGE_MPC::addInputStageCost(DM R)
    {   
        // create symbolic variables stage cost at t=k
        MX u_cost = MX::sym("u_cost", n_controls_);

        for (int k = 0; k < N_; k++)
        {   
            // Create Stage Cost term for input trajectory (u(k))R(u(k))           
            u_cost = U_(all,k);
            
            costFunction_ += mtimes(mtimes(u_cost.T(),R),u_cost);
        }

        // update the NLP
        opti_.minimize(costFunction_);
        std::cout<< "added Input Stage Cost" << std::endl;
    }


    void SPONGE_MPC::addTerminalCost(DM P)
    {
        // Create Terminal Cost term
        MX x_cost_ = MX::sym("x_cost", n_states_);

        // difference to last value of reference state trajectory (x(t+N)-x_ref(t+N)) P (x(t+N)-x_ref(t+N))
        x_cost_ = X_(all,N_) - X_GOAL_(all,N_); 

        // Add difference weighted with P to the Cost Function
        costFunction_ += mtimes(mtimes(x_cost_.T(),P),x_cost_);
            
        // update  NLP
        opti_.minimize(costFunction_);
        std::cout<< "added Terminal Cost" << std::endl;
    }


    void SPONGE_MPC::addStateDifferenceCost(DM dQ)
    {
        // Create Difference State Cost term
        MX d_x_cost = MX::sym("d_x_cost", n_states_);

        for (int k = 1; k < N_+1; k++)
        {
            // difference of x(k) - x(k-1)
            d_x_cost = X_(all,k) - X_(all,k-1);
            // Add difference weighted with dQ to the Cost Function  
            costFunction_ += mtimes(mtimes(d_x_cost.T(),dQ),d_x_cost);    
        }

        // update NLP
        opti_.minimize(costFunction_);
        std::cout<< "added State Difference Cost" << std::endl;
    }


    void SPONGE_MPC::addInputDifferenceCost(DM dR)
    {
        MX d_u_cost = MX::sym("d_u_cost", n_controls_);

        for (int k = 1; k < N_; k++)
        {
            // difference of u(k) - u(k-1)
            d_u_cost = U_(all,k) - U_(all,k-1);
            // Add difference weighted with dR to the Cost Function
            costFunction_ += mtimes(mtimes(d_u_cost.T(),dR),d_u_cost);
        }

        // update NLP
        opti_.minimize(costFunction_);
        std::cout<< "added Input Difference Cost" << std::endl;
    }


    void SPONGE_MPC::addMeanInputCost(DM R_p_mean, DM p_mean_scale)
    {
        // Create Cost term
        MX p_mean_cost_ = MX::sym("p_mean_cost", n_states_);

        // difference from computet inputs to mean input
        for (int i = 0; i < 2 * n_states_; i += 2)
        {   
            p_mean_cost_ = (U_(i,all) + U_(i+1,all) - p_mean_scale);
            // Add to cost function
            costFunction_ += mtimes(mtimes(p_mean_cost_,R_p_mean),p_mean_cost_.T());
        }
            
        // update  NLP
        opti_.minimize(costFunction_);
        std::cout<< "added Input Mean Cost" << std::endl;
    }


    DM SPONGE_MPC::ComputeControlInput(DM x_meas, DM x_ref, DM h_k_minus_1)
    {
        // Set initial measured and reference state trajectory and hidden states
        opti_.set_value(X_GOAL_, x_ref);
        opti_.set_value(H_K_, h_k_minus_1);
        opti_.set_value(X_K_, x_meas);

        // actual solve
        OptiSol solution_ = opti_.solve();   

        x_opt = solution_.value(X_);
        u_opt = solution_.value(U_);

        std::cout<<"x_meas: "<< x_meas << std::endl;
        std::cout<< "x_ref: "<< x_ref << std::endl;

        // For warmstarting update initial guess for next iteration with shifted solutions
        X0_ = reshape(vertcat(x_opt(Slice(n_states_, (N_+1)*n_states_)), x_opt(Slice(N_*n_states_,(N_+1)*n_states_))), n_states_, N_+1);
        U0_ = reshape(vertcat(u_opt(Slice(n_controls_, N_*n_controls_)), u_opt(Slice((N_-1)*n_controls_,N_*n_controls_))), n_controls_, N_);
        
        opti_.set_initial(X_,X0_);
        opti_.set_initial(U_,U0_);


        // return first predicted input value
        return solution_.value(U_(Slice(0,n_controls_)));
    }

    } // namespace sponge

