#include <sstream>
#include <pthread.h>
#include <semaphore.h>
#include "ros/ros.h"
#include <actionlib/server/simple_action_server.h>
#include "std_msgs/Float64.h"
#include "std_msgs/String.h"
#include <dynamic_reconfigure/server.h>
#include <pcu_sl_interface/modelConfig.h>
#include <chrono>
#include <thread>
#include <typeinfo>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "SL_func.h"
#include "sponge_mpc/mpc_step.h"

class SLNode
{
protected:
    sem_t sem_out_newdata_; 
    pthread_mutex_t mut_out_; 
    pthread_mutex_t mut_in_; 
    pthread_mutexattr_t  mutattr_prioinherit_; 
    pthread_mutex_t mut_state_;
    SL_OUT_type sl_out_buffer_;
    SL_IN_type sl_in_buffer_;
    bool is_stopped_;
    ros::NodeHandle nh_;
    ros::AsyncSpinner spinner_;
    ros::ServiceClient mpc_client_;
    int n_akt=5;
    double ctrl_state;
    int data_horizon;
    sponge_mpc::mpc_step mpc_srv;
    int len_q_des_deg;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_MPC,end_time_MPC;
    std::chrono::duration<double> duration_MPC;
public:
    SLNode():
        is_stopped_(false),
        spinner_(1)
{
        ROS_ASSERT(!sem_init(&sem_out_newdata_, 0, 0));
        ROS_ASSERT(!pthread_mutexattr_init(&mutattr_prioinherit_));
        ROS_ASSERT(!pthread_mutexattr_setprotocol(&mutattr_prioinherit_, PTHREAD_PRIO_INHERIT));
        ROS_ASSERT(!pthread_mutex_init(&mut_in_, &mutattr_prioinherit_));
        ROS_ASSERT(!pthread_mutex_init(&mut_out_, &mutattr_prioinherit_));
        ROS_ASSERT(!pthread_mutex_init(&mut_state_, NULL));
        nh_ = ros::NodeHandle("~");
        mpc_client_= nh_.serviceClient<sponge_mpc::mpc_step>("/mpc_step_service");
        spinner_.start();

}

    ~SLNode(void)
    {
    }

    void SL_io_func(SL_OUT_type* sl_out, SL_IN_type* sl_in){
        int val = 1;
        pthread_mutex_lock(&mut_in_);
        memcpy((void *)sl_in, (void *)&sl_in_buffer_, sizeof(SL_IN_type));
        pthread_mutex_unlock(&mut_in_);
        sem_getvalue(&sem_out_newdata_, &val);
        if(val == 0){
            pthread_mutex_lock(&mut_out_);
            memcpy((void *)&sl_out_buffer_, (void *)sl_out, sizeof(SL_OUT_type));
            pthread_mutex_unlock(&mut_out_); 
            sem_post(&sem_out_newdata_);
        }
    }

    void read_buffer(){
        pthread_mutex_lock(&mut_out_);
        mpc_srv.request.q_des_deg=std::vector<double>();
        mpc_srv.request.qd_des_degs=std::vector<double>();
        mpc_srv.request.q_deg=std::vector<double>();
        mpc_srv.request.qd_degs=std::vector<double>();
        data_horizon=static_cast<int>(sl_out_buffer_.q_des_deg[0]);
        for(int i=0;i<n_akt;++i){
          mpc_srv.request.q_deg.push_back(sl_out_buffer_.q_deg[i]);
          mpc_srv.request.qd_degs.push_back(sl_out_buffer_.qd_degs[i]);
        }
        len_q_des_deg=static_cast<int>(sl_out_buffer_.q_des_deg[0]*n_akt+1);
        for(int i=0;i<len_q_des_deg;++i){
          mpc_srv.request.q_des_deg.push_back(sl_out_buffer_.q_des_deg[i]);
          mpc_srv.request.qd_des_degs.push_back(sl_out_buffer_.qd_des_degs[i]);
          } 
        ctrl_state = sl_out_buffer_.ctrl_state;
        pthread_mutex_unlock(&mut_out_);
    }

    void write_buffer(double t_sol_s){
        pthread_mutex_lock(&mut_in_);
        for(int i=0;i<n_akt;++i){
            sl_in_buffer_.p_d_bar[i*2]=mpc_srv.response.p_des_bar[i*2];
            sl_in_buffer_.p_d_bar[i*2+1]=mpc_srv.response.p_des_bar[i*2+1];
        }
        sl_in_buffer_.t_sol_s=t_sol_s;
        pthread_mutex_unlock(&mut_in_);
    }
    
    void run(){   
        sem_wait(&sem_out_newdata_);
        while (ros::ok() && !is_stopped_)
        {
            read_buffer();
            start_time_MPC = std::chrono::high_resolution_clock::now();
            if (ctrl_state>0)
                {
                if (mpc_client_.call(mpc_srv))
                    {
                    end_time_MPC= std::chrono::high_resolution_clock::now();
                    duration_MPC = end_time_MPC - start_time_MPC;
                    ROS_INFO("MPC step successful (solution time: %.2fs)!",duration_MPC.count());
                    write_buffer(duration_MPC.count());   
                    }
                else
                    {ROS_ERROR("MPC call failed");}
                }
            sem_wait(&sem_out_newdata_);
            ros::spinOnce();
        }
    }

    void stop(){
        is_stopped_ = true;
        sem_post(&sem_out_newdata_);
    }
};

pthread_t ros_thread;
void *ros_thread_fn(void* arg);
sem_t sem_init_finished;
SLNode* slNode;
void SL_io_func(SL_OUT_type* sl_out, SL_IN_type* sl_in){
    slNode->SL_io_func(sl_out, sl_in);
}

void SL_start_func(){
    std::cout << "Creating ROS-Node Thread!" << std::endl;
    ROS_ASSERT(!sem_init(&sem_init_finished, 0, 0)); 
    pthread_create(&ros_thread, NULL, &ros_thread_fn, NULL);
    sem_wait(&sem_init_finished); 
}

void SL_terminate_func(){
    std::cout << "Terminating ROS-Node Thread!" << std::endl;
    slNode->stop();
    pthread_join(ros_thread, NULL);
}

void *ros_thread_fn(void* arg)
{
    int argc = 0;
    char **argv = NULL;

    struct sched_param param = {};
    param.sched_priority = 90;
    if (sched_setscheduler (0, SCHED_RR, &param) == -1){/*empty*/};
    ros::init(argc, argv, "SL_RT_CORE", ros::init_options::NoSigintHandler);
    slNode = new SLNode();
    sem_post(&sem_init_finished);
    slNode->run();
    delete(slNode);
    ros::shutdown();
    ROS_INFO("OUT!");
    return 0;
}
