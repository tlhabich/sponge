#include "models/RobotSystem.h"

using namespace sponge;

RobotSystem::RobotSystem(){
    n_states_ = N_STATES;
    n_controls_ = N_INPUTS;
    n_outputs_ = N_INPUTS;
    n_hidden_states_ = HIDDEN_STATES;
    n_hidden_layers_ = HIDDEN_LAYERS;

    // load system parameters from config file
    load_net_params();

    // // create system GRU
    createGRU();
}

RobotSystem::~RobotSystem(){}

void RobotSystem::setStateConstraints(DM x_ub, DM x_lb)
{
  x_ub_ = x_ub;
  x_lb_ = x_lb;
}

void RobotSystem::setInputConstraints(DM u_ub, DM u_lb)
{
  u_ub_ = u_ub;
  u_lb_ = u_lb;
}

// sigmoid function
MX RobotSystem::sigmoid(const MX& input) {
  return 1.0 / (1.0 + exp(-input));
}

void RobotSystem::createGRU()
{
    std::cout<< "RobotSystem - createGRU" << std::endl;
    // symbolic states
    MX x = MX::sym("x",N_STATES); // state vector
    MX states = x;

    // symbolic inputs
    MX u = MX::sym("u",N_INPUTS); // virtual input to robot system
    MX controls = u;

    MX x_t = vertcat(states, controls);

    // symbolic hidden states
    MX h_t_minus_1 = MX::sym("h_t_minus_1",HIDDEN_STATES,HIDDEN_LAYERS);  // hidden states for every GRU layer
    std::cout<<"h_t_minus_1: "<<h_t_minus_1.size1()<<"    "<<h_t_minus_1.size2()<< std::endl;
    MX h_t0_minus_1 = h_t_minus_1;

    // define hidden layer 0 of GRU
    MX r_t0 = sigmoid(mtimes(w_ir_l0_,x_t)  + b_ir_l0_.T() + mtimes(w_hr_l0_,h_t0_minus_1) + b_hr_l0_.T());       // Reset Gate
    MX z_t0 = sigmoid(mtimes(w_iz_l0_,x_t) + b_iz_l0_.T() + mtimes(w_hz_l0_,h_t0_minus_1) + b_hz_l0_.T());        // Update Gate
    MX n_t0 = tanh(mtimes(w_in_l0_,x_t) + b_in_l0_.T() + r_t0 * (mtimes(w_hn_l0_,h_t0_minus_1) + b_hn_l0_.T()));  // New Gate
    MX h_t0 = (1 - z_t0) * n_t0 + z_t0 * h_t0_minus_1; // New Hidden State

    // define linear output layer of GRU
    MX x_t_plus_1 = mtimes(w_linout_,h_t0) + b_linout_.T();
    MX h_t = h_t0;

    GRU_ = Function ("gru", {states,controls,h_t_minus_1}, {x_t_plus_1});

}

void RobotSystem::load_net_params()
{
  std::cout<< "RobotSystem - load_net_params" << std::endl;
  // function to load neural network weights from yaml file
  YAML::Node config_nn = YAML::LoadFile("app_interface/ros_install/include/models/sys_params.yaml"); // Prüfstand

  // GRU net parameters
  int rows = config_nn["w_ir_l0"].size();
  int cols = config_nn["w_ir_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              w_ir_l0_(i,j) = config_nn["w_ir_l0"][i][j].as<double>();
          }
      }

  // w_iz_l0_
  rows = config_nn["w_iz_l0"].size();
  cols = config_nn["w_iz_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              w_iz_l0_(i,j) = config_nn["w_iz_l0"][i][j].as<double>();
          }
      }

  // w_in_l0_
  rows = config_nn["w_in_l0"].size();
  cols = config_nn["w_in_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              w_in_l0_(i,j) = config_nn["w_in_l0"][i][j].as<double>();
          }
      }

  // b_ir_l0_
  rows = config_nn["b_ir_l0"].size();
  cols = config_nn["b_ir_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              b_ir_l0_(i,j) = config_nn["b_ir_l0"][i][j].as<double>();
          }
      }

  // b_iz_l0_
  rows = config_nn["b_iz_l0"].size();
  cols = config_nn["b_iz_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              b_iz_l0_(i,j) = config_nn["b_iz_l0"][i][j].as<double>();
          }
      }

  // b_in_l0_
  rows = config_nn["b_in_l0"].size();
  cols = config_nn["b_in_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              b_in_l0_(i,j) = config_nn["b_in_l0"][i][j].as<double>();
          }
      }

  // w_hr_l0_
  rows = config_nn["w_hr_l0"].size();
  cols = config_nn["w_hr_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              w_hr_l0_(i,j) = config_nn["w_hr_l0"][i][j].as<double>();
          }
      }

  // w_hz_l0_
  rows = config_nn["w_hz_l0"].size();
  cols = config_nn["w_hz_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              w_hz_l0_(i,j) = config_nn["w_hz_l0"][i][j].as<double>();
          }
      }

  // w_hn_l0_
  rows = config_nn["w_hn_l0"].size();
  cols = config_nn["w_hn_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              w_hn_l0_(i,j) = config_nn["w_hn_l0"][i][j].as<double>();
          }
      }

  // b_hr_l0_
  rows = config_nn["b_hr_l0"].size();
  cols = config_nn["b_hr_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              b_hr_l0_(i,j) = config_nn["b_hr_l0"][i][j].as<double>();
          }
      }

  // b_hz_l0_
  rows = config_nn["b_hz_l0"].size();
  cols = config_nn["b_hz_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              b_hz_l0_(i,j) = config_nn["b_hz_l0"][i][j].as<double>();
          }
      }

  // b_hn_l0_
  rows = config_nn["b_hn_l0"].size();
  cols = config_nn["b_hn_l0"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              b_hn_l0_(i,j) = config_nn["b_hn_l0"][i][j].as<double>();
          }
      }

  // w_linout_
  rows = config_nn["weight_linout"].size();
  cols = config_nn["weight_linout"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              w_linout_(i,j) = config_nn["weight_linout"][i][j].as<double>();
          }
      }
    
  // b_linout
  rows = config_nn["bias_linout"].size();
  cols = config_nn["bias_linout"][0].size();
  for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              b_linout_(i,j) = config_nn["bias_linout"][i][j].as<double>();
          }
      }

}

void RobotSystem::display_net_params()
{
    std::cout << "RobotSystem - display_net_params" << std::endl;

    std::cout << "w_ir_l0_: " << std::endl;
    std::cout << w_ir_l0_ << std::endl;

}