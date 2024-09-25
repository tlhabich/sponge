#include <models/ModelBase.h>

ModelBase::ModelBase(){}
ModelBase::~ModelBase(){}

int ModelBase::get_n_controls(){return n_controls_;};
int ModelBase::get_n_states(){return n_states_;};
int ModelBase::get_n_outputs(){return n_outputs_;};
int ModelBase::get_n_hidden_states(){return n_hidden_states_;};
int ModelBase::get_n_hidden_layers(){return n_hidden_layers_;};

// DM ModelBase::getRefStateAndInput(DM y_ref, double dt_mpc) {};

SX ModelBase::eval(const SX &state, const SX &control, const SX &h_t_minus_1)
{
    // function of GRU neural net evaluated with casadis Symbolic expression SX
    SX state_next_ = GRU_((vector<SX>){state, control, h_t_minus_1}).at(0);   

    return state_next_;
}

DM ModelBase::eval(const DM &state, const DM &control, const DM &h_t_minus_1)
{
    // function of GRU neural net evaluated with casadis Numeric expression DM
    DM state_next_ = GRU_((vector<DM>){state, control, h_t_minus_1}).at(0); 
       
    return state_next_;
}

MX ModelBase::eval(const MX &state, const MX &control, const MX &h_t_minus_1)
{
    // function of GRU neural net evaluated with casadis Numeric expression MX
    MX state_next_ = GRU_((vector<MX>){state, control, h_t_minus_1}).at(0); 
       
    return state_next_;
}

