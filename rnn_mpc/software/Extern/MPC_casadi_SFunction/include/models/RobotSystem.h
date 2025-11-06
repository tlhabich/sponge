// #pragma once

#ifndef ROBOT_SYSTEM
#define ROBOT_SYSTEM

#include <models/ModelBase.h>
#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>

#include <sl_function/sl_mpc.h>

#include <yaml-cpp/yaml.h>

using namespace std;
using namespace casadi;

namespace sponge
{
  
class RobotSystem : public  ModelBase
{
private:
  // GRU net parameters
  DM w_ir_l0_ = DM::zeros(HIDDEN_STATES,N_STATES+N_INPUTS);
  DM w_iz_l0_ = DM::zeros(HIDDEN_STATES,N_STATES+N_INPUTS);
  DM w_in_l0_ = DM::zeros(HIDDEN_STATES,N_STATES+N_INPUTS);

  DM b_ir_l0_ = DM::zeros(1,HIDDEN_STATES);
  DM b_iz_l0_ = DM::zeros(1,HIDDEN_STATES);
  DM b_in_l0_ = DM::zeros(1,HIDDEN_STATES);

  DM w_hr_l0_ = DM::zeros(HIDDEN_STATES,HIDDEN_STATES);
  DM w_hz_l0_ = DM::zeros(HIDDEN_STATES,HIDDEN_STATES);
  DM w_hn_l0_ = DM::zeros(HIDDEN_STATES,HIDDEN_STATES);

  DM b_hr_l0_ = DM::zeros(1,HIDDEN_STATES);
  DM b_hz_l0_ = DM::zeros(1,HIDDEN_STATES);
  DM b_hn_l0_ = DM::zeros(1,HIDDEN_STATES);

  DM w_linout_ = DM::zeros(N_STATES,HIDDEN_STATES);
  DM b_linout_ = DM::zeros(1,N_STATES);

  // lower-/upper bounds of state and input constraints
  DM x_ub_, x_lb_, u_ub_, u_lb_;

  // to access a whole row or colon in a casadi matrix MX, MX or MX
  Slice all;

  void createGRU();

public:

  RobotSystem();  
  ~RobotSystem();

  void setStateConstraints(DM x_ub, DM x_lb);
  void setInputConstraints(DM u_ub, DM u_lb);

  // declaration of the sigmoid function
  MX sigmoid(const MX& input);

  void load_net_params();
  void display_net_params();
};

} // namespace sponge

#endif // ROBOT_SYSTEM

