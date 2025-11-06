#include <models/ModelBase.h>

ModelBase::ModelBase(){}
ModelBase::~ModelBase(){}

// Getters for model dimensions
int ModelBase::get_n_controls(){return n_controls_;};
int ModelBase::get_n_states(){return n_states_;};
int ModelBase::get_n_outputs(){return n_outputs_;};
int ModelBase::get_n_hidden_states(){return n_hidden_states_;};
int ModelBase::get_n_hidden_layers(){return n_hidden_layers_;};

// DM ModelBase::getRefStateAndInput(DM y_ref, double dt_mpc) {};

// Evaluate GRU neural network with CasADi symbolic expressions
SX ModelBase::eval(const SX &state, const SX &control, const SX &h_t_minus_1)
{
    SX state_next_ = GRU_((vector<SX>){state, control, h_t_minus_1}).at(0);   
    return state_next_;
}

// Evaluate GRU neural network with CasADi numeric expressions
DM ModelBase::eval(const DM &state, const DM &control, const DM &h_t_minus_1)
{
    DM state_next_ = GRU_((vector<DM>){state, control, h_t_minus_1}).at(0); 
    return state_next_;
}

// Evaluate GRU neural network with CasADi matrix expressions
MX ModelBase::eval(const MX &state, const MX &control, const MX &h_t_minus_1)
{
    MX state_next_ = GRU_((vector<MX>){state, control, h_t_minus_1}).at(0); 
    return state_next_;
}

