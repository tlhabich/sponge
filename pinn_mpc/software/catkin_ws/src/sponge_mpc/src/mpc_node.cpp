#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <fstream>
#include "sponge_mpc/mpc_step.h"
#include <cmath> 
#include "../include/json.hpp"
#include <casadi/casadi.hpp> //https://github.com/casadi/casadi/wiki/InstallationLinux
#include <string>
#include <cmath>
#include <typeinfo>
#include <chrono>
using json = nlohmann::json;
using namespace casadi;
using namespace std;

class MPCServer {
public:
    MPCServer(nlohmann::json jsonData) {
        service = nh.advertiseService("mpc_step_service", &MPCServer::handle_mpc_step,this);
        // read ROS params
        nh.getParam("/N",N);
        nh.getParam("/N_u",N_u);
        nh.getParam("/p_max_bar",p_max_bar_saturation);
        nh.getParam("/Q_pos",Q_pos);
        nh.getParam("/Q_pos_terminal",Q_pos_terminal);
        nh.getParam("/Qd_pos",Qd_pos);
        nh.getParam("/Qd_pos_terminal",Qd_pos_terminal);
        nh.getParam("/R_p",R_p);
        nh.getParam("/beta_test_deg",beta_test_deg);
        nh.getParam("/mE_test_g",mE_test_g);
        nh.getParam("/addPosStageCost",addPosStageCost);
        nh.getParam("/addInputStageCost",addInputStageCost);
        nh.getParam("/addPosTerminalCost",addPosTerminalCost);
        nh.getParam("/addVelTerminalCost",addVelTerminalCost);
        nh.getParam("/addVelStageCost",addVelStageCost);
        nh.getParam("/multiple_shooting_flag",multiple_shooting_flag);
        p_max_Pa=p_max_bar_saturation*pow(10, 5);
        mE_test_kg=mE_test_g*pow(10, -3);

        // read PINN params
        n_neurons=jsonData["n_neurons"];
        n_hidden=jsonData["n_hidden"];
        input_dim=jsonData["input_dim"];
        output_dim=jsonData["output_dim"];
        factor_downsampling=jsonData["factor_downsampling"];
        n_akt=jsonData["n_akt"];
        udim=jsonData["udim"];
        xdim=jsonData["xdim"];
        lr_init=jsonData["lr_init"];
        betamax=jsonData["betamax"];
        betamin=jsonData["betamin"];
        mEmax=jsonData["mEmax"];
        mEmin=jsonData["mEmin"];
        pmax=jsonData["pmax"];
        qmax=jsonData["qmax"];
        qdmax=jsonData["qdmax"];
        t_sample=jsonData["t_sample"];
        T=jsonData["T"];
        ddpinn_flag=jsonData["ddpinn_flag"];
        ddpinn_props_ansatz=jsonData["ddpinn_props"]["ansatz"];
        ddpinn_props_n_ansatz=jsonData["ddpinn_props"]["n_ansatz"];
        xdim_ansatz=ddpinn_props_n_ansatz*xdim;
        m_x["q"]=jsonData["m_x"]["q"];
        m_x["qd"]=jsonData["m_x"]["qd"];
        m_x["p"]=jsonData["m_x"]["p"];
        m_x["t"]=jsonData["m_x"]["t"];
        m_x["beta"]=jsonData["m_x"]["beta"];
        m_x["mE"]=jsonData["m_x"]["mE"];
        m_x["tau"]=jsonData["m_x"]["tau"];
        b_x["q"]=jsonData["b_x"]["q"];
        b_x["qd"]=jsonData["b_x"]["qd"];
        b_x["p"]=jsonData["b_x"]["p"];
        b_x["t"]=jsonData["b_x"]["t"];
        b_x["beta"]=jsonData["b_x"]["beta"];
        b_x["mE"]=jsonData["b_x"]["mE"];
        b_x["tau"]=jsonData["b_x"]["tau"];
        beta_test_scaled=this->normalizeVar(this->deg2rad(beta_test_deg), m_x["beta"], b_x["beta"], true);
        mE_test_scaled=this->normalizeVar(mE_test_kg, m_x["mE"], b_x["mE"], true);
        t_sample_scaled=this->normalizeVar(t_sample, m_x["t"], b_x["t"], true);
        pmax_scaled=this->normalizeVar(p_max_Pa, m_x["p"], b_x["p"], true);
        pmin_scaled=this->normalizeVar(0, m_x["p"], b_x["p"], true);
        if (ddpinn_flag){
            t_sample_scaled=t_sample_scaled+1; // t_raw=0s equals t_scaled=-1
        } 
        for (int i_layer = 0; i_layer < n_hidden+2; ++i_layer) {
            // weights
            vector<vector<double>> w_matrix;
            string w_key="w"+to_string(i_layer);
            for (const auto& row : jsonData["weights"][w_key]) {
                w_matrix.push_back(row.get<vector<double>>());
            }
            nn_params[w_key] = w_matrix;
            // biases
            vector<vector<double>> b_matrix;
            string b_key="b"+to_string(i_layer);
            for (const auto& row : jsonData["biases"][b_key]) {
                b_matrix.push_back(row.get<vector<double>>());
            }
            nn_params[b_key] = b_matrix;
        }
        w0=DM(nn_params["w0"]).T();
        w1=DM(nn_params["w1"]).T();
        w2=DM(nn_params["w2"]).T();
        w3=DM(nn_params["w3"]).T();
        b0=DM(nn_params["b0"]).T();
        b1=DM(nn_params["b1"]).T();
        b2=DM(nn_params["b2"]).T();
        b3=DM(nn_params["b3"]).T();

        pos_weighting=DM::eye(n_akt);
        vel_weighting=DM::eye(n_akt);
        u_weighting=DM::eye(2*n_akt);
        ROS_INFO("Parameters (PINN and ROS) initialized");
               
        string prefix_code = ros::package::getPath("sponge_mpc") + "/code_gen/";

        if(true){
            // COMPILATION
            opti_compile = Opti();
            options["hessian_approximation"] = "limited-memory";
            opti_compile.solver("ipopt", coptions, options);

            U_ = opti_compile.variable(2*n_akt, N_u); // control trajectory
            X_GOAL_ = opti_compile.parameter(n_akt, N); // desired path
            XD_GOAL_ = opti_compile.parameter(n_akt, N); //desired velocity
            X_K_ = opti_compile.parameter(2*n_akt, 1); // current measured values           
            if(multiple_shooting_flag){
                    X_ = opti_compile.variable(2*n_akt, N+1); // state trajectory
                    opti_compile.subject_to(X_(all,0) == X_K_);  // initial constraint
                }
            costFunction_ = 0;

            // dynamic constraint
            for (int k = 0; k < N; ++k) {
                if(k==0){
                    X_K_next_ = this->predictFNN(X_K_,U_(all,0));
                }
                else if(k<N_u){
                    X_K_next_ = this->predictFNN(X_K_next_,U_(all,k));
                }
                else{
                    X_K_next_ = this->predictFNN(X_K_next_,U_(all,N_u-1));
                }
                if(multiple_shooting_flag){
                    opti_compile.subject_to(X_(all, k + 1) == X_K_next_);
                }
                if (addPosStageCost && k<N-1){
                    q_cost =  X_K_next_(Slice(0,n_akt)) - X_GOAL_(all,k);
                    costFunction_ += MX::mtimes(MX::mtimes(q_cost.T(),pos_weighting*Q_pos),q_cost);
                }
                if (addPosTerminalCost && k==N-1){
                    q_cost_terminal = X_K_next_(Slice(0,n_akt)) - X_GOAL_(all,k);
                    costFunction_ += mtimes(mtimes(q_cost_terminal.T(),pos_weighting*Q_pos_terminal),q_cost_terminal);
                }        
                if (addVelStageCost && k<N-1){
                    qd_cost =  X_K_next_(Slice(n_akt,2*n_akt)) - XD_GOAL_(all,k);
                    costFunction_ += MX::mtimes(MX::mtimes(qd_cost.T(),vel_weighting*Qd_pos),qd_cost);
                }
                if (addVelTerminalCost && k==N-1){
                    qd_cost_terminal = X_K_next_(Slice(n_akt,2*n_akt)) - XD_GOAL_(all,k);
                    costFunction_ += mtimes(mtimes(qd_cost_terminal.T(),vel_weighting*Qd_pos_terminal),qd_cost_terminal);
                }
                if (addInputStageCost && k<N_u){
                    u_cost = U_(all,k);
                    costFunction_ += MX::mtimes(MX::mtimes(u_cost.T(),u_weighting*R_p),u_cost);
                }
            }

            // input constraint
            for (int i = 0; i < 2*n_akt; i++)
            {
                opti_compile.subject_to(pmin_scaled <= U_(i,all) <= pmax_scaled);
            }
            
            opti_compile.minimize(costFunction_);

            // compile
            opti_compile.set_value(X_K_,DM::zeros(2*n_akt, 1)); 
            opti_compile.set_value(X_GOAL_,DM::zeros(n_akt, N));
            opti_compile.set_value(XD_GOAL_,DM::zeros(n_akt, N));
            OptiSol sol_ = opti_compile.solve();            
            Dict opts = Dict();
            opts["cpp"] = false;
            opts["with_header"] = true;

            // IPOPT
            Function nlp_grad_f = opti_compile.debug().casadi_solver().get_function("nlp_grad_f");
            CodeGenerator myCodeGen = CodeGenerator("nlp_grad_f.c", opts);
            myCodeGen.add(nlp_grad_f);
            myCodeGen.generate(prefix_code);
            cout<<"Compiling nlp_grad_f..."<<endl;
            string compile_command = "gcc -fPIC -shared -O3 " + 
                prefix_code + "nlp_grad_f.c -o " +
                prefix_code + "nlp_grad_f.so";
            int compile_flag = system(compile_command.c_str());
            casadi_assert(compile_flag==0, "Compilation failed!");
            cout << "Compilation of nlp_grad_f successed!" << endl;

            Function nlp_jac_g = opti_compile.debug().casadi_solver().get_function("nlp_jac_g");
            myCodeGen = CodeGenerator("nlp_jac_g.c", opts);
            myCodeGen.add(nlp_jac_g);
            myCodeGen.generate(prefix_code);
            cout<<"Compiling nlp_jac_g..."<<endl;
            compile_command = "gcc -fPIC -shared -O3 " + 
                prefix_code + "nlp_jac_g.c -o " +
                prefix_code + "nlp_jac_g.so";
            compile_flag = system(compile_command.c_str());
            casadi_assert(compile_flag==0, "Compilation failed!");
            cout << "Compilation of nlp_jac_g successed!" << endl;
        }

        opti_ = Opti();
        options["print_level"] = 0;
        coptions["print_time"] = false;
        options["hessian_approximation"] = "limited-memory";
        Function nlp_grad_f_obj= external("nlp_grad_f",prefix_code+"nlp_grad_f.so");
        Function nlp_jac_g_obj= external("nlp_jac_g",prefix_code+"nlp_jac_g.so");
        coptions["grad_f"]=nlp_grad_f_obj;
        coptions["jac_g"]=nlp_jac_g_obj;
        options["tol"] = pow(10,-1);
        opti_.solver("ipopt",coptions, options);
        
        U_ = opti_.variable(2*n_akt, N_u); // control trajectory
        X_GOAL_ = opti_.parameter(n_akt, N); // desired path
        XD_GOAL_ = opti_.parameter(n_akt, N); //desired velocity
        X_K_ = opti_.parameter(2*n_akt, 1); // current measured values
        
        if(multiple_shooting_flag){
                X_ = opti_.variable(2*n_akt, N+1); // state trajectory
                opti_.subject_to(X_(all,0) == X_K_);  // initial constraint
            }

        costFunction_ = 0;
       
        // dynamic constraint
        for (int k = 0; k < N; ++k) {
            if(k==0){
                X_K_next_ = this->predictFNN(X_K_,U_(all,0));
            }
            else if(k<N_u){
                X_K_next_ = this->predictFNN(X_K_next_,U_(all,k));
            }
            else{
                X_K_next_ = this->predictFNN(X_K_next_,U_(all,N_u-1));
            }
            
            if(multiple_shooting_flag){
                opti_.subject_to(X_(all, k + 1) == X_K_next_);
            }

            
            if (addPosStageCost && k<N-1){
                q_cost =  X_K_next_(Slice(0,n_akt)) - X_GOAL_(all,k);
                costFunction_ += MX::mtimes(MX::mtimes(q_cost.T(),pos_weighting*Q_pos),q_cost);
            }
            
            if (addPosTerminalCost && k==N-1){
                q_cost_terminal = X_K_next_(Slice(0,n_akt)) - X_GOAL_(all,k);
                costFunction_ += mtimes(mtimes(q_cost_terminal.T(),pos_weighting*Q_pos_terminal),q_cost_terminal);
            }
            
            if (addVelStageCost && k<N-1){
                    qd_cost =  X_K_next_(Slice(n_akt,2*n_akt)) - XD_GOAL_(all,k);
                    costFunction_ += MX::mtimes(MX::mtimes(qd_cost.T(),vel_weighting*Qd_pos),qd_cost);
                }
            if (addVelTerminalCost && k==N-1){
                qd_cost_terminal = X_K_next_(Slice(n_akt,2*n_akt)) - XD_GOAL_(all,k);
                costFunction_ += mtimes(mtimes(qd_cost_terminal.T(),vel_weighting*Qd_pos_terminal),qd_cost_terminal);
            }
            
            if (addInputStageCost && k<N_u){
                u_cost = U_(all,k);
                costFunction_ += MX::mtimes(MX::mtimes(u_cost.T(),u_weighting*R_p),u_cost);
            }   
        }

        // input constraint
        for (int i = 0; i < 2*n_akt; i++)
        {
            opti_.subject_to(pmin_scaled <= U_(i,all) <= pmax_scaled);
        }
        opti_.minimize(costFunction_);
        std::cout<<"Opti: "<< opti_ << std::endl;
        ROS_INFO("Casadi initialized");
        }

    bool handle_mpc_step(sponge_mpc::mpc_step::Request &req, sponge_mpc::mpc_step::Response &res) {
        // read values
        q_deg=DM(req.q_deg);
        qd_degs=DM(req.qd_degs);
        // get desired values in data horizon (specified in Matlab)
        data_horizon=to_int(req.q_des_deg[0]);
        Q_des_deg_full=DM::reshape(DM(req.q_des_deg)(Slice(1,data_horizon*n_akt+1),0),data_horizon,n_akt).T();
        Qd_des_degs_full=DM::reshape(DM(req.qd_des_degs)(Slice(1,data_horizon*n_akt+1),0),data_horizon,n_akt).T();
        q_des_scaled=this->normalizeVar(this->deg2rad(Q_des_deg_full(all,Slice(1,N+1))), m_x["q"], b_x["q"], true);
        qd_des_scaled=this->normalizeVar(this->deg2rad(Qd_des_degs_full(all,Slice(1,N+1))), m_x["qd"], b_x["qd"], true);
        q_scaled=this->normalizeVar(this->deg2rad(q_deg), m_x["q"], b_x["q"], true);
        qd_scaled=this->normalizeVar(this->deg2rad(qd_degs), m_x["qd"], b_x["qd"], true);
        xk_meas_scaled=DM::vertcat({q_scaled,qd_scaled});
        opti_.set_value(X_K_,xk_meas_scaled); 
        opti_.set_value(X_GOAL_,q_des_scaled);
        opti_.set_value(XD_GOAL_,qd_des_scaled);
        OptiSol sol_ = opti_.solve();
        u_opt = sol_.value(U_);
        U0_ = reshape(horzcat(u_opt(all,Slice(1,N_u)), u_opt(all,N_u-1)), 2*n_akt, N_u);
        opti_.set_initial(U_,U0_);

        if(multiple_shooting_flag){
            x_opt = sol_.value(X_); 
            X0_ = reshape(horzcat(x_opt(all,Slice(1, N+1)), x_opt(all,N)), 2*n_akt, N+1); //Slice(all,des_col+1)
            opti_.set_initial(X_,X0_);
        }
        p_des_bar=this->normalizeVar(u_opt(all,0), m_x["p"], b_x["p"], false)*pow(10, -5);
        ROS_INFO("Solution found.");
        res.p_des_bar=p_des_bar.nonzeros();
        return true;
    }
    double rad2deg(double angle_rad){
        return angle_rad*180/M_PI;
    }
    DM rad2deg(DM angle_rad){
        return angle_rad*180/M_PI;
    }
    double deg2rad(double angle_deg){
        return angle_deg*M_PI/180;
    }
    DM deg2rad(DM angle_deg){
        return angle_deg*M_PI/180;
    }
    double normalizeVar(double var,double m_var,double b_var,bool flag){
        if (flag){
            // normalize
            return var*m_var+b_var;
        }
        else{
            // renormalize
            return (var-b_var)*1/m_var;  
        }   
    }
    DM normalizeVar(DM var,double m_var,double b_var,bool flag){
        if (flag){
            // normalize
            return var*m_var+b_var;
        }
        else{
            // renormalize
            return (var-b_var)*1/m_var;  
        }   
    }
    
    MX predictFNN(MX x,MX u){
        if (this->ddpinn_flag){
            X_out=MX::vertcat({x,u,beta_test_scaled,mE_test_scaled}).T();
        }   
        else{
            X_out=MX::vertcat({x,u,beta_test_scaled,mE_test_scaled,t_sample_scaled}).T();
        }
        X_out=MX::tanh(MX::mtimes(X_out,w0)+b0);
        X_out=MX::tanh(MX::mtimes(X_out,w1)+b1);
        X_out=MX::tanh(MX::mtimes(X_out,w2)+b2);
        X_out=MX::mtimes(X_out,w3)+b3;

        if (this->ddpinn_flag){            
            bt_plus_c=X_out(0,Slice(xdim_ansatz,2*xdim_ansatz))*t_sample_scaled+X_out(0,Slice(2*xdim_ansatz,3*xdim_ansatz));
            exp_neg_dt=MX::exp(-X_out(0,Slice(3*xdim_ansatz,4*xdim_ansatz))*t_sample_scaled);
            act_g_bt_plus_c=MX::sin(bt_plus_c);
            g_out=X_out(0,Slice(0,xdim_ansatz))*(act_g_bt_plus_c*exp_neg_dt-MX::sin(X_out(0,Slice(2*xdim_ansatz,3*xdim_ansatz))));
            g_out_sum=MX::sum1(MX::reshape(g_out,ddpinn_props_n_ansatz,xdim));
            X_out=(x+g_out_sum.T()).T();
        }
        return X_out.T();
    }
        
    

private:
    ros::NodeHandle nh;
    ros::ServiceServer service;
    int N,N_u,n_neurons,n_hidden,input_dim,output_dim,factor_downsampling,n_akt,udim,xdim,ddpinn_props_n_ansatz,xdim_ansatz,data_horizon;
    double p_max_bar_saturation,p_max_Pa,Q_pos,Qd_pos,Qd_pos_terminal,Q_pos_terminal,R_p,beta_test_deg,mE_test_g,mE_test_kg,t_sample_scaled,
    lr_init,betamax,betamin,mEmax,mEmin,pmax,qmax,qdmax,t_sample,T,beta_test_scaled,mE_test_scaled,pmax_scaled,pmin_scaled,qmax_scaled,qdmax_scaled;
    bool addPosStageCost,addInputStageCost,addPosTerminalCost,addVelTerminalCost,addVelStageCost,ddpinn_flag,multiple_shooting_flag;
    string ddpinn_props_ansatz;
    map<string,double> m_x,b_x;
    map<string,vector<vector<double>>> nn_params;
    Opti opti_,opti_compile;
    Dict options, coptions;
    MX U_, X_GOAL_,XD_GOAL_, X_K_,costFunction_,X_K_next_,X_,q_cost,q_cost_terminal,qd_cost,qd_cost_terminal,u_cost;
    MX X_out,bt_plus_c,exp_neg_dt,act_g_bt_plus_c,g_out,g_out_sum;
    Slice all;
    DM q_deg, qd_degs, p_bar,Q_des_deg_full,Qd_des_degs_full,q_des_scaled,qd_des_scaled,q_scaled,qd_scaled,xk_meas_scaled,x_opt,u_opt,sol_,X0_,U0_,p_des_bar,vel_weighting,pos_weighting,u_weighting;
    DM w0,w1,w2,w3,b0,b1,b2,b3;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time,end_time,sub_time;

};

int main(int argc, char **argv) {
    ros::init(argc, argv, "mpc_node");
    ros::NodeHandle nh;
    // read PINN data
    string pinn_name;
    nh.getParam("/pinn_name",pinn_name);
    string package_path = ros::package::getPath("sponge_mpc");   
    ifstream file(package_path+"/cfg/"+pinn_name+".json");
    if (!file.is_open()) {
        ROS_ERROR("Error: Could not open the file!");
        return 1;
    }
    else {
        ROS_INFO("Found PINN model: %s",pinn_name.c_str());
    }

    nlohmann::json jsonData;
    file >> jsonData;
    file.close();
    MPCServer server(jsonData);
    ROS_INFO("Ready for MPC");
    ros::spin();
    return 0;
}


