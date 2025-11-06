// #pragma once

#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace casadi;


class ModelBase
{
protected:

  Function GRU_;
  int n_states_, n_controls_, n_outputs_;

  int n_hidden_states_, n_hidden_layers_;

public:

  ModelBase();
  ~ModelBase();
    
    // Evaluate GRU network with different CasADi types
    SX eval(const SX &state, const SX &control, const SX &h_t_minus_1);
    DM eval(const DM &state, const DM &control, const DM &h_t_minus_1);
    MX eval(const MX &state, const MX &control, const MX &h_t_minus_1);

    // Getters for model dimensions
    int get_n_states();
    int get_n_controls();
    int get_n_outputs();
    int get_n_hidden_states();
    int get_n_hidden_layers();
};


// } // namespace sponge

#endif // MODEL_BASE_H