#include <iostream>
#include <math.h> 
#include <chrono>
#include <thread>
#include <fstream>

#include <casadi/casadi.hpp>
#include <yaml-cpp/yaml.h>
#include <models/RobotSystem.h>
#include <mpc_lib/sponge_mpc.h> 
#include <sl_function/sl_mpc.h>

// include path of MPC-params
const std::string mpc_param_config_file = "app_interface/ros_install/include/models/mpc_params.yaml";

using namespace casadi;
using namespace std;
using namespace sponge;


RobotSystem *dynamics_mpc;
SPONGE_MPC *sponge_mpc;


int n_states, n_controls, n_outputs, pred_horizon, n_hidden_states, n_hidden_layers;

DM x_meas;      // initial (measured) state
DM x_ref;       // reference state
DM h_t_minus_1; // hidden states
DM u;           // computed input

double dt_mpc;  // sample time of MPC controller

bool first_run = true;

std::chrono::high_resolution_clock::time_point start_time;


void get_mpc_params(const string config_file, MPC_PARAM_type* params)
{   
  // read yaml file 
  YAML::Node config = YAML::LoadFile(config_file);
  std::cout << "YAML file loaded, dt_mpc:: "<< config["dt_mpc"] << std::endl;
  
  std::cout<< "Using MPC with:"
  << "\ndt_mpc = "<< params->dt_mpc
  << "\nprediction horizon N = "<< params->pred_horizon
  << std::endl;
  
}

// StartFcnSpec — Function that the S-function calls when it begins execution, specified as a character vector or string.
void SL_start_mpc_func(MPC_PARAM_type* params)
{  
    // Load parameters from config file
    // Note: Parameters sometimes don't parse correctly through s-function block
    get_mpc_params(mpc_param_config_file, params);

    // Initialize robot system
    dynamics_mpc = new RobotSystem();
    // dynamics_mpc->display_net_params(); // to check whether the parameters have been loaded correctly
    
    // Get system dimensions
    n_states = dynamics_mpc->get_n_states();
    n_controls = dynamics_mpc->get_n_controls(); 
    n_outputs = dynamics_mpc->get_n_outputs();
    n_hidden_states = dynamics_mpc->get_n_hidden_states();
    n_hidden_layers = dynamics_mpc->get_n_hidden_layers();

    pred_horizon = params->pred_horizon;
    dt_mpc = params->dt_mpc;

    // Initialize state vectors
    x_meas = DM::zeros(n_states);
    x_ref = DM::zeros(n_states,pred_horizon+1);
    h_t_minus_1 = DM::zeros(n_hidden_states,n_hidden_layers);

    // Print MPC configuration
    std::cout << "MPC Configuration:"
              << "\nSample time: " << dt_mpc
              << "\nPrediction horizon: " << pred_horizon
              << std::endl;

    // Initialzize weighting matrices and contraint vectors
    DM P = DM::zeros(n_states,n_states);
    DM Q = DM::zeros(n_states,n_states);
    DM dR = DM::zeros(n_controls,n_controls);
    DM R = DM::zeros(n_controls,n_controls);
    DM x_ub = DM::zeros(n_states, 1);
    DM x_lb = DM::zeros(n_states, 1);
    DM u_ub = DM::zeros(n_controls, 1);
    DM u_lb = DM::zeros(n_controls, 1);

    DM R_p_mean = DM::zeros(pred_horizon, pred_horizon);
    DM p_mean_scale = DM::zeros(1);

    // write weighting matrices and contraint vectors from f-function param to casadi variables
    for (int i = 0; i < n_states; i++)
    {
      Q(i,i) = params->Q[i];
      P(i,i) = params->P[i];
      x_ub(i) = params->x_ub[i];
      x_lb(i) = params->x_lb[i];
      R(i,i) = params->R[i];
      dR(i,i) = params->dR[i];
    }

    for (int i = 0; i < n_controls; i++)
    {
      R(i,i) = params->R[i];
      dR(i,i) = params->dR[i];
      u_ub(i) = params->u_ub[i];
      u_lb(i) = params->u_lb[i];
    }

    for (int i = 0; i < pred_horizon; i++)
    {
      R_p_mean(i,i) = 0.02;
    }
    p_mean_scale = -0.1;

    // create object of the MPC controller for the given system
    sponge_mpc = new SPONGE_MPC(dynamics_mpc, pred_horizon, dt_mpc, dt_mpc);

    // add cost terms and constraints to MPC controller  
    sponge_mpc->addStateStageCost(Q);
    sponge_mpc->addInputStageCost(R);
    sponge_mpc->addTerminalCost(P);
    sponge_mpc->addInputDifferenceCost(dR);
    sponge_mpc->addStateConstraints(x_ub, x_lb);
    sponge_mpc->addInputConstraints(u_ub, u_lb);
    sponge_mpc->addMeanInputCost(R_p_mean, p_mean_scale);
    // sponge_mpc->addStateDifferenceCost(dQ);
    

    std::cout<< "MPC controller configured" << std::endl;
}

void SL_io_mpc_func(MPC_IN_type* inports, MPC_OUT_type* outports){

  start_time = std::chrono::high_resolution_clock::now();

  if (inports->on_off != 1){
    std::cout<< "MPC is not turned on..." << std::endl;
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return;
  }
  if (isnan(inports->state[0])){
    std::cout<< "input state is nan" << std::endl;
    return;
  }
  if (isinf(inports->state[0])){
    std::cout<< "input state is inf" << std::endl;
    return;
  }

  // Read current state from input ports
  for (int i = 0; i < n_states; i++){ 
    x_meas(i) = inports->state[i];
  }

  // Read state trajectory from input ports
 int k = 0;
for (int j = 0; j < pred_horizon+1; j++) {
    for (int i = 0; i < n_states; i++) {
        x_ref(i, j) = inports->ref[0][k];
        k++;
    }
}

  // Read hidden states from input ports
  for (int i = 0; i < n_hidden_states; i++){
    for (int j = 0; j < n_hidden_layers; j++){
      h_t_minus_1(i,j) = inports->h_t_minus_1[i][j];
    }
  }

  // compute next input by solving the MPC problem
   u = sponge_mpc->ComputeControlInput(x_meas, x_ref, h_t_minus_1);

  // write computed input in output port 
  for (int i = 0; i < n_controls; i++){
    outports->mv[i] = (double)u(i);
  }

  std::cout<< "[S-FuncBlock] execution time " <<  
  1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-start_time).count()
  << std::endl;
  
}

void SL_terminate_mpc_func(){
    std::cout<<"MPC Terminated"<<std::endl;
    
    delete sponge_mpc;
    delete dynamics_mpc;
}