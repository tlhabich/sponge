# -*- coding: utf-8 -*-
"""
Classes/methods/functions for PINN_main.py
@author: TLH
"""

import torch
import tempfile
import numpy as np
from pandas import read_csv
import copy
import os
import random
from pathlib import Path
import pathlib
from datetime import datetime
from timeit import default_timer as timer
import PINN_physics as pinnphys
import math
from scipy.stats import qmc
import pickle
import ray
import json
import pdb
from ray.train import Checkpoint
from ray import tune, train

class pinn_cl():
    def __init__(self,cwd):
        self.retrain_network=False,"pinn_obj","adam",None # (retrain_flag, nn_name, new_optimizer,new_scheduler_min_lr)
        self.n_collo=int(100e3)# number collocation points
        self.ic_size=.2# use n_collo*ic_size points for initial condition loss (only for PINC)
        self.n_data=0# number real-world data points (<0 -> use all, =0 -> use nothing, >0 cut dataset)
        self.accelerate_training=True # physics loss: calculate state-space model without gradients -> considerably faster
        self.test_mode=False# only for fast testing        
        self.path_train_data=Path(cwd,"measurement_data","beta0deg_mE0g_ident.csv")
        self.path_test_data=Path(cwd,"measurement_data","beta0deg_mE0g_ramp.csv")
        self.f_sample_data=50# sampling frequency Hz
        self.factor_downsampling=1# t_sample=factor_downsampling*1/f_sample_data -> larger step size for PINN
        self.hpo_flag=False # HPO (hyperparameter optimization with ray-tune)
        self.scheduler_min_lr=5e-5 # minimal learning rate
        # network structure during standalone training
        self.n_neurons=100# number of neurons in hidden layers
        self.n_hidden=2# number of hidden layers
        self.lr_init=5e-4# initial learning rate
        self.num_epochs=2001# will be overwritten for HPO
    
    def initParams(self):
        #init remaining params
        if self.hpo_flag:
            if self.pinn_flag:
                # HPO config PINN
                self.hpo_config={
                   'n_neurons': tune.randint(50,200), 
                   'n_hidden': tune.choice([1,2,3]), 
                   'lr_init': tune.uniform(5e-5,5e-3),
                   'n_ansatz': tune.randint(10,50)
                           }
            else:
                # HPO config RNN
                self.hpo_config={
                   'n_neurons': tune.randint(50,200), 
                   'n_hidden': tune.choice([1,2,3]), 
                   'lr_init': tune.uniform(5e-5,5e-3),
                   'dropout_GRU': tune.uniform(0.2, 0.5)
                           }
            self.num_epochs=1001
            self.ray_cpus=32#CPUs for HPO
            self.ray_samples=100#number HPO trials
            self.asha_grace_period=500#PINN
            # self.asha_grace_period=100#RNN
            self.asha_reduction_factor=2
        self.pinn_flag=True#PINN or black-box model
        self.blackbox_mode="FNN" #"FNN" to learn delta_x or "RNN" for GRUs
        self.batch_size=512# batch training (in general: training could be accelerated when everything, e.g. also physics-loss calculation with ODE and differentiation, is calculated batch-wise!)
        self.plot_interval=25#export results all x epocchs 
        self.ddpinn_flag=True #if True, use DD-PINN instead of classical PINC
        self.ddpinn_props={
            "n_ansatz":50, # number of ansatz function
            "ansatz":"damped" # definition of the used ansatz
            }
        self.dropout_GRU=0.25
        if self.pinn_flag:
            self.scheduler_patience=50#ReduceLROnPlateau
        else:
            self.scheduler_patience=10#ReduceLROnPlateau
            self.n_collo=0
            self.n_ic=0
        
        # other parameters (changed less frequently)
        self.test_dur_s=30 #number of datapoints during test
        self.sample_normal_dist_x=True#sample multivariate normal samples with mean at 0 (only for xdim states instead of LHS)
        self.loss_weighting="naive"#"naive" ("lra" and "softadapt" also implemented, but not fully tested!)
        self.alpha_weighting=.2#moving-average weighting for lra and softadapt 
        self.loss_weighting_interval=1 #for lra and softadapt 
        self.softAdapt_naive_flag=True
        self.g_KSW=torch.tensor([-9.81,0,0])#gravitational vector at beta=0
        self.n_akt=5 #number of actuators
        self.mpc_horizon=0#reset state with measurement similar during MPC (0 -> disabled)
        self.scheduler_factor=0.5# ReduceLROnPlateau
        self.act_str="Tanh" #activation function: Tanh, Sigmoid, SELU, GELU
        self.first_order_dyn=False#if True, qd is not a state -> this simplifies the ODE (but is only a rough approximation!)
        self.resampling_period=251#after x epochs resampling of ic and collocation points
        self.optimizer_type="adam" #lbfgs or adam (lbfgs very slow and not fully tested!)
        self.with_rc=True#model Coulomb friction
        self.split_size=0.7# trainining/validation data split
        self.rotate_y=True #rotate with beta around y? Otherwise around z
        self.coldim=2#additional inputs for collocation: beta,mE
        # collocation/sampling limits
        self.betamax=1.25*pinnphys.deg2rad(90)
        self.betamin=0
        self.mEmax=1.25*0.2
        self.mEmin=0
        self.pmax=1.25*0.7*10**5
        self.qmax=1.25*pinnphys.deg2rad(25)
        self.qdmax=1.25*pinnphys.deg2rad(30)
        self.t_sample=self.factor_downsampling*1/self.f_sample_data
        self.T=self.t_sample*1.25
        
        self.plot_stages=3#number of mE and beta variations during test
        if self.test_mode:
            self.asha_grace_period=2
            self.ray_samples=4
            self.ray_cpus=4
            self.n_collo=250
            self.batch_size=50
            self.test_dur_s=2
            self.plot_interval=5
            self.num_epochs=10
            self.plot_stages=1
        self.loss_fn=torch.nn.MSELoss()
        self.ray_interval=self.plot_interval#save model all x epochs 
        self.neglect_p_dyn=True#assume p=p_des (False also implemented, but not fully tested!)
        self.xdim=4*self.n_akt #q,qd,p1,p2
        if self.neglect_p_dyn:
            self.xdim=self.xdim-2*self.n_akt 
        if self.first_order_dyn:
            self.xdim=self.xdim-self.n_akt 
        self.udim=2*self.n_akt #pd1,pd2
        self.input_dim = self.udim+self.xdim+self.coldim+1#+1 for t (dimension will be reduced later, if black-box model is trained)
        self.output_dim=self.xdim
        #initial condition points
        if self.pinn_flag:
            if self.ddpinn_flag:
                self.n_ic=0#no ic points necessary
            else:
                self.n_ic=int(self.n_collo*self.ic_size)
        self.eta_p=pinnphys.GetParams(self.n_akt,0)["eta_p"] #eta_p=A_p*r_p
        self.qd_c=pinnphys.GetParams(self.n_akt,0)["qd_c"] #Coulomb parameter
        # normalization stuff
        self.xmax={'q':self.qmax,'qd':self.qdmax,'p':self.pmax,'t':self.T,'beta':self.betamax,'mE':self.mEmax,'tau':self.eta_p*self.pmax}
        self.xmin={'q':-self.qmax,'qd':-self.qdmax,'p':0,'t':0,'beta':self.betamin,'mE':self.mEmin,'tau':-self.eta_p*self.pmax}
        self.m_x={}
        self.b_x={}
        for lbl in ["q","qd","p","t","beta","mE","tau"]:
            self.m_x[lbl]=2/(self.xmax[lbl]-self.xmin[lbl])
            self.b_x[lbl]=-2*self.xmin[lbl]/(self.xmax[lbl]-self.xmin[lbl])-1
        self.t_sample_scaled=normalizeVar(self.t_sample, self.m_x["t"], self.b_x["t"], True)
        self.n_losses=4#collo_qd,collo_qdd,ic,data (deactive losses included: for example, one could split the loss for qd and qdd which is currently disabled)
        self.img_format="png"#png or pdf
        self.test_dur=self.test_dur_s*self.f_sample_data
                
    def loadTrainData(self):
        if self.n_data>0 and self.pinn_flag:
            self.X_raw_train,self.Y_raw_train,self.indices,self.beta_train_raw,self.mE_train_raw,t_sample_true=loadData(self.path_train_data,
                                                                                                        0,self.n_akt,self.xdim,
                                                                                                        self.udim,self.first_order_dyn,
                                                                                                        self.neglect_p_dyn,
                                                                                                        self.factor_downsampling)
            if self.n_data<self.X_raw_train.shape[0]:
                downsample_indices=np.linspace(0,self.X_raw_train.shape[0]-1,self.n_data,dtype=int)
                self.X_raw_train=self.X_raw_train[downsample_indices,:]
                self.Y_raw_train=self.Y_raw_train[downsample_indices,:]
        else:
            self.X_raw_train,self.Y_raw_train,self.indices,self.beta_train_raw,self.mE_train_raw,t_sample_true=loadData(self.path_train_data,
                                                                                                        self.n_data,self.n_akt,self.xdim,
                                                                                                        self.udim,self.first_order_dyn,
                                                                                                        self.neglect_p_dyn,
                                                                                                        self.factor_downsampling)
        if self.first_order_dyn: print("1st order dynamics (simpler but only rough approximation)!")
        else: print("2nd order dynamics!")
        print("PINN flag: "+str(self.pinn_flag)+"\t\tDD-PINN flag: "+str(self.ddpinn_flag))
        print("Train dataset with beta="+str(round(pinnphys.rad2deg(self.beta_train_raw),2))+"deg and mE="+str(self.mE_train_raw)+"kg")
        if round(t_sample_true,4)!=self.t_sample:
            raise ValueError("t_sample of downsampled data does not equal self.t_sample!")
        else:
            print("Train for t_sample="+str(round(self.t_sample,2))+"s (T="+str(round(self.T,2))+"s)")
        self.U_raw_train=self.X_raw_train[:,self.indices["u"]]
        if self.n_data!=0:
            self.n_data=self.X_raw_train.shape[0]
            print("Number of datapoints in train dataset: "+str(self.n_data))
            #use experimental data
            data_points_raw=self.addColVar()
            data_points=normalizeXY(data_points_raw,self.neglect_p_dyn,self.first_order_dyn,self.indices,self.m_x,self.b_x,
                                    self.coldim,self.input_dim,True,self.xdim,self.udim)        
        else:
            data_points=[]
        self.data_points=data_points
        
    def loadTestData(self):
        self.X_raw_test,self.Y_raw_test,temp1,self.beta_test_raw,self.mE_test_raw,temp2=loadData(self.path_test_data,self.test_dur,self.n_akt,self.xdim, self.udim,self.first_order_dyn,
                             self.neglect_p_dyn)
        print("Test dataset with beta="+str(round(pinnphys.rad2deg(self.beta_test_raw),2))+"deg and mE="+str(self.mE_test_raw)+"kg")
        self.U_raw_test=self.X_raw_test[:,self.indices["u"]]
        self.U_scaled_test=normalizeVar(self.U_raw_test, self.m_x["p"], self.b_x["p"], True)

    def createDataloaderNoPinn(self):
        if self.pinn_flag==False:
            self.noPinnDL()
        else:
            self.train_DL_noPinn=None
            self.val_DL_noPinn=None
            
    def createFolder(self,cwd):
        try: 
            os.mkdir("Standalone_results")
        except:
            temp=0
        today = datetime.now()
        date = today.strftime("%d_%m_%Y__%H_%M")
        self.base_folder = Path(str(cwd) + str('/Standalone_results/') + str(date))
        try:
            os.mkdir(self.base_folder)
        except:
            print(str(self.base_folder)+" already exists")
            
    def initTraining(self):
        if self.pinn_flag:
            self.model=FNN(self.input_dim,self.n_hidden,self.n_neurons,self.output_dim,self.act_str,self.ddpinn_flag,self.ddpinn_props)
        else:
            if self.blackbox_mode=="RNN":
                self.model = GRU(self.input_dim,self.n_hidden,self.n_neurons,self.output_dim,self.dropout_GRU)
            elif self.blackbox_mode=="FNN":
                self.model = FNN_blackbox(self.input_dim, self.n_hidden, self.n_neurons, self.output_dim,self.act_str)
            else:
                raise ValueError("blackbox_mode not supported")
        self.optimizer=initOptimizer(self.optimizer_type,self.model.parameters(),self.lr_init)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = self.scheduler_factor, 
                                                                    patience = self.scheduler_patience,threshold=0,
                                                                    min_lr=self.scheduler_min_lr)
        self.train_DL_Pinn=None
        self.val_DL_Pinn=None
        self.train_loss_history={"phys_p":np.ones(self.num_epochs),"phys_qd":np.ones(self.num_epochs),
                                 "phys_qdd":np.ones(self.num_epochs),"data":np.ones(self.num_epochs),
                                 "ic":np.ones(self.num_epochs),"phys_p_scaled":np.ones(self.num_epochs),
                                 "phys_qd_scaled":np.ones(self.num_epochs),"phys_qdd_scaled":np.ones(self.num_epochs),
                                 "data_scaled":np.ones(self.num_epochs),"ic_scaled":np.ones(self.num_epochs),
                                 "total":np.ones(self.num_epochs)}
        self.val_loss_history={"phys_p":np.ones(self.num_epochs),"phys_qd":np.ones(self.num_epochs),
                                 "phys_qdd":np.ones(self.num_epochs),"data":np.ones(self.num_epochs),
                                 "ic":np.ones(self.num_epochs),"phys_p_scaled":np.ones(self.num_epochs),
                                 "phys_qd_scaled":np.ones(self.num_epochs),"phys_qdd_scaled":np.ones(self.num_epochs),
                                 "data_scaled":np.ones(self.num_epochs),"ic_scaled":np.ones(self.num_epochs),
                                 "total":np.ones(self.num_epochs)}
        self.loss_lam_history=np.ones((self.num_epochs,self.n_losses))*np.nan
        self.learning_rate_history=np.ones((self.num_epochs,1))*np.nan
        self.loss_lam_final=np.ones([self.n_losses])#used losses in first epoch
        self.epoch_time_history_s=np.ones((self.num_epochs,1))*np.nan
        
    def train(self):
        for epoch in range(self.num_epochs):
            epoch_start_timer = timer()
            self.trainOneEpoch(epoch)
            if self.nan_flag:
                print("End of training!")
                if self.hpo_flag: train.report({"mean_loss":10000,"iterations":epoch})
                break
                            
            train_loss=self.train_loss_history["total"][epoch]
            val_loss=self.val_loss_history["total"][epoch]
            epoch_end_timer = timer()
            dur=epoch_end_timer - epoch_start_timer
            self.epoch_time_history_s[epoch]=dur
            print("Epoch #"+str(epoch+1)+" took "+str(round(dur,1))+" seconds -> "+str(round(dur*(self.num_epochs-(epoch+1))/60**2,1))+
                  " hours of training left!\n"+
                  "Training loss: "+str(round(train_loss,5))+"\tValidation loss: "+str(round(val_loss,5))+"\nCurrent learning rate: "+str(self.optimizer.param_groups[0]["lr"])+"\n###")
            
            if epoch>self.scheduler_patience:
                #warm-up in the first #scheduler-patience epochs
                self.scheduler.step(val_loss)
            self.learning_rate_history[epoch]=self.optimizer.param_groups[0]["lr"]            
                        
            if (epoch%self.plot_interval==0 and epoch!=0) or epoch==self.num_epochs-1:
                #export stats
                if self.hpo_flag:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        self.plotPerformance(temp_checkpoint_dir,epoch)
                        train.report({"mean_loss":val_loss,"iterations":epoch}, checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))     
                else:
                    path = Path(self.base_folder,"Epoche_" + str(epoch))
                    try:os.mkdir(path)
                    except:donothing=0
                    self.plotPerformance(path,epoch)
                    
            elif self.hpo_flag:
                train.report({"mean_loss":val_loss,"iterations":epoch})   
           
    def trainOneEpoch(self,epoch):
        if epoch%self.resampling_period==0 and self.pinn_flag:
            #Sample collocation points and initial condition points und merge with data_points
            print("Resampling collocation and initial condition points")
            self.sampleColloInit(self.output_dim)
        if self.pinn_flag==False:
            train_DL=self.train_DL_noPinn
            val_DL=self.val_DL_noPinn
        else:
            train_DL=self.train_DL_Pinn
            val_DL=self.val_DL_Pinn
        for train_mode in [True,False]:
            if train_mode:
                dataloader=train_DL
            else:
                dataloader=val_DL
            if train_mode==False and (self.pinn_flag==False or self.ddpinn_flag):
                torch.set_grad_enabled(False)
                self.model.eval()
            else:
                torch.set_grad_enabled(True)#necessary for PINC due to autograd (physics loss)
                self.model.train()
            data_loss_sum,ic_loss_sum,phys_loss_p_sum,phys_loss_qd_sum,phys_loss_qdd_sum,loss_sum=0,0,0,0,0,0
            data_loss_sum_scaled,ic_loss_sum_scaled,phys_loss_p_sum_scaled,phys_loss_qd_sum_scaled,phys_loss_qdd_sum_scaled=0,0,0,0,0
            if train_mode and self.loss_weighting=="lra":
                i_batch_LRA=random.randint(0,len(dataloader)-1)
            # loop through batches (training could be accelerated when everything, e.g. also physics-loss calculation with ODE and differentiation, is calculated batch-wise! Currently: point-wise)
            for i, (X_daten, Y_daten) in enumerate(dataloader):
                batch_size=X_daten.shape[1]
                X_daten = torch.reshape(X_daten, (batch_size, self.input_dim))
                Y_daten = torch.reshape(Y_daten, (batch_size, self.output_dim+1))
                if self.pinn_flag==False:
                    if i==0:
                        x_k = torch.reshape(X_daten[0,:self.xdim], (1,1,self.xdim))
                        if self.blackbox_mode=="RNN": h_k=torch.zeros(self.model.n_hidden, 1, self.model.n_neurons)
                    else:
                        # truncated backpropagation-through-time
                        x_k=x_k.detach()
                        if self.blackbox_mode=="RNN": h_k = h_k.detach()
                def closure():
                    if self.pinn_flag==False:
                        nonlocal x_k
                        if self.blackbox_mode=="RNN": nonlocal h_k
                        u_batch = torch.zeros(X_daten.shape[0], self.udim)
                        u_batch[:,:] = X_daten[:,self.xdim:]
                        Y_pred = torch.zeros((Y_daten.shape[0],self.xdim))
                        Y_ground_truth = torch.zeros(Y_pred.shape)
                    if train_mode: self.optimizer.zero_grad()
                    point_type=Y_daten[:,-1].int().numpy()
                    phys_loss_p,phys_loss_qd,phys_loss_qdd,data_loss,ic_loss=torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)
                    n_data_batch,n_ic_batch,n_collo_batch=0,0,0
                    for k in range(batch_size):
                        if self.pinn_flag:
                            X_step = torch.reshape(X_daten[k,:], (1,1,self.input_dim))
                        if point_type[k]==1:
                            n_collo_batch+=1
                            #collocation point
                            t_step=torch.reshape(X_step[0,0,-1],(1,1,1))
                            t_step=t_step.detach()
                            t_step.requires_grad=True
                            
                            temp=torch.reshape(X_step[0,0,:-1],(1,1,self.input_dim-1))
                            X_step_t = torch.cat([temp, t_step], dim=2)
                            
                            if self.ddpinn_flag:
                                Y_dach,dY_dachdt = self.model(X_step_t,True)
                            else:
                                Y_dach= self.model(X_step_t)
                                t_step.retain_grad()
                                try: t_step.grad.zero_()
                                except: donothing=0
                            
                            #nonlinear dynamics used as physics loss
                            dqdt_scaled=torch.empty([self.n_akt,1])
                            dqdt_raw=torch.empty([self.n_akt,1])
                            if self.neglect_p_dyn==False:
                                dpdt_raw=torch.empty([2*self.n_akt,1])
                                dpdt_scaled=torch.empty([2*self.n_akt,1])
                            if self.first_order_dyn==False:
                                dqddt_scaled=torch.empty([self.n_akt,1])
                                dqddt_raw=torch.empty([self.n_akt,1])
                            for row in range(self.xdim):
                                if self.ddpinn_flag:
                                    # autograd not necessary
                                    temp=dY_dachdt[0,0,row].reshape(1,1,1)
                                else:
                                    # autograd
                                    temp,=torch.autograd.grad(Y_dach[0,0,row],t_step,retain_graph=True,create_graph=True)
                                if row<self.n_akt:
                                    dqdt_scaled[row,:]=temp[0,0,0]
                                    dqdt_raw[row,:]=temp[0,0,0]*self.m_x["t"]/self.m_x["q"]
                                elif self.first_order_dyn==False and row<2*self.n_akt:
                                    dqddt_scaled[row-self.n_akt,:]=temp[0,0,0]
            
                                    dqddt_raw[row-self.n_akt,:]=temp[0,0,0]*self.m_x["t"]/self.m_x["qd"]
                                else:
                                    if self.first_order_dyn: temp_ind=self.n_akt
                                    else: temp_ind=2*self.n_akt
                                    dpdt_scaled[row-temp_ind,:]=temp[0,0,0]
                                    dpdt_raw[row-temp_ind,:]=temp[0,0,0]*self.m_x["t"]/self.m_x["p"]
                            
                            #renormalization
                            p_des_raw=normalizeVar(X_step[0,0,self.indices["u"]], self.m_x["p"], self.b_x["p"], False)
                            if self.neglect_p_dyn:
                                #p=p_des
                                p_dach_raw=normalizeVar(X_step[0,0,self.indices["u"]], self.m_x["p"], self.b_x["p"], False).detach().numpy()
                            else:
                                #pressure dynamics
                                p_dach_raw=normalizeVar(Y_dach[0,0,self.indices["p"]], self.m_x["p"], self.b_x["p"], False)
                            
                            q_dach_raw=normalizeVar(Y_dach[0,0,self.indices["q"]], self.m_x["q"], self.b_x["q"], False)
                            if self.first_order_dyn==False:
                                qd_dach_raw=normalizeVar(Y_dach[0,0,self.indices["qd"]], self.m_x["qd"], self.b_x["qd"], False)
                                qd_dach_scaled=Y_dach[0,0,self.indices["qd"]]  
                                                       
                            #col var
                            if self.coldim==1:
                                beta_raw=normalizeVar(X_step[0,0,-2], self.m_x["beta"], self.b_x["beta"], False)
                                mE_raw=torch.tensor(self.mE_train_raw)
                            elif self.coldim==2:
                                beta_raw=normalizeVar(X_step[0,0,-3], self.m_x["beta"], self.b_x["beta"], False)
                                mE_raw=normalizeVar(X_step[0,0,-2], self.m_x["mE"], self.b_x["mE"], False)
                            else:
                                #no additional col vars -> training with beta_raw and mE_raw from train data
                                beta_raw=torch.tensor(self.beta_train_raw)
                                mE_raw=torch.tensor(self.mE_train_raw)
                            g_col_raw=pinnphys.calc_g_KS0(beta_raw, self.g_KSW,self.rotate_y,True)
                                
                            lam_dqdt=self.m_x["q"]/self.m_x["t"]
                            lam_dqddt=self.m_x["qd"]/self.m_x["t"]
                            if self.first_order_dyn:
                                if self.with_rc:
                                    #dqdt implicit in ODE -> add coulomb to dqdt from autograd
                                    dqdt_raw=pinnphys.addCoulomb(dqdt_raw,q_dach_raw,self.n_akt)
                                    dqdt_scaled=torch.mul(dqdt_raw,lam_dqdt)
                            
                                #right side of first order ODE
                                dqdt_maple_raw=pinnphys.calcRedFdyn_HC(q_dach_raw,p_dach_raw,self.n_akt,g_col_raw,mE_raw,False,self.accelerate_training)
                                #normalize maple dgl
                                dqdt_maple_scaled=torch.mul(dqdt_maple_raw,lam_dqdt)
                                
                                #gradients
                                loss_qd=self.loss_fn(dqdt_scaled,dqdt_maple_scaled)
                                loss_qdd=torch.tensor(0.0)
                            else:
                                right_side_ODE_raw,Mass=pinnphys.calcImplDyn(q_dach_raw,qd_dach_raw,p_dach_raw,self.n_akt,g_col_raw,mE_raw,self.accelerate_training)
                                Mqdd_raw=torch.matmul(Mass,dqddt_raw)
                                
                                #ONE phys loss
                                dqdt_Mdqddt_raw=torch.vstack((dqdt_raw,Mqdd_raw))
                                qd_right_side_raw=torch.vstack((qd_dach_raw.reshape((self.n_akt,1)),right_side_ODE_raw))
                                
                                if self.accelerate_training:
                                    loss_qd=self.loss_fn(dqdt_Mdqddt_raw,qd_right_side_raw.detach())
                                else:
                                    loss_qd=self.loss_fn(dqdt_Mdqddt_raw,qd_right_side_raw)
                                loss_qdd=torch.tensor(0.0) #currently only placeholder
    
                            if self.neglect_p_dyn==False:
                                raise ValueError("training with pressure dynamics not implemented")
                            phys_loss_qd+=loss_qd
                            phys_loss_qdd+=loss_qdd
                        else:#data point or ic point
                            if self.pinn_flag:
                                Y_ground_truth=Y_daten[k,:-1]
                                Y_dach = self.model(X_step)
                                Y_dach=Y_dach.reshape(self.output_dim)
                                if point_type[k]==2:
                                    #data point
                                    n_data_batch+=1
                                    temp_loss=self.loss_fn(Y_dach.reshape(self.xdim),Y_ground_truth)
                                    data_loss += temp_loss
                                elif point_type[k]==3:
                                    #ic point
                                    n_ic_batch+=1
                                    temp_loss=self.loss_fn(Y_dach,Y_ground_truth)
                                    ic_loss += temp_loss
                            else:
                                #no pinn
                                u_k=u_batch[k,:]# true inputs
                                X_step = torch.hstack((x_k.reshape(self.xdim), u_k.reshape(self.udim)))
                                X_step = X_step.reshape(1, 1, self.input_dim)
                                if self.blackbox_mode=="RNN": Y_dach, h_k = self.model(X_step, h_k)
                                else: Y_dach = self.model(X_step)
                                Y_pred[k,:] = torch.reshape(Y_dach, (1,self.xdim))
                                Y_ground_truth[k,:]=Y_daten[k,:-1]
                                #for next iteration
                                x_k=Y_dach
                    
                    if self.pinn_flag==False:
                        data_loss = self.loss_fn(Y_pred, Y_ground_truth)
                    else:
                        #mean losses for batch
                        if n_data_batch>0: data_loss=data_loss/n_data_batch
                        if n_collo_batch>0:
                            phys_loss_p=phys_loss_p/n_collo_batch
                            phys_loss_qd=phys_loss_qd/n_collo_batch
                            phys_loss_qdd=phys_loss_qdd/n_collo_batch
                        if n_ic_batch>0:
                            ic_loss=ic_loss/n_ic_batch
                        
                    #loss scaling (LRA one time per epoch for random batch)
                    if train_mode and self.loss_weighting=="lra" and i==i_batch_LRA and self.pinn_flag and epoch>0 and epoch%self.loss_weighting_interval==0:
                        #LRA only for one batch
                        self.LRA(data_loss,ic_loss,phys_loss_qd)
                        
                    #save
                    if train_mode and i==len(dataloader)-1: self.loss_lam_history[epoch,:]=self.loss_lam_final
                    
                    #Scale loss terms
                    loss=self.loss_lam_final[0]*ic_loss+self.loss_lam_final[1]*phys_loss_qd+self.loss_lam_final[2]*data_loss
                    if self.first_order_dyn==False:
                        loss+=self.loss_lam_final[3]*phys_loss_qdd
                    
                    #save losses
                    self.closure_losses=[ic_loss.detach().numpy(),phys_loss_qd.detach().numpy(),data_loss.detach().numpy(),phys_loss_qdd.detach().numpy(),loss.detach().numpy()]
                    
                    self.nan_flag=False
                    if torch.isnan(loss):
                        print(f'loss is NaN!')
                        self.nan_flag=True
                        return
                    
                    if train_mode: 
                        loss.backward()
                        # feedback for lbfgs training
                        if self.optimizer_type=="lbfgs":
                            try: self.lbfgs_iter+=1
                            except: self.lbfgs_iter=1
                            print("################")
                            print("LBFGS iteration: "+str(self.lbfgs_iter))
                            print("ic loss: "+str(self.closure_losses[0]))
                            print("phys_qd loss: "+str(self.closure_losses[1]))
                            if self.n_data>0: print("data loss: "+str(self.closure_losses[2]))
                            print("total loss: "+str(self.closure_losses[4]))
                    return loss
                
                #pptimize
                if train_mode:
                    self.optimizer.step(closure)                    
                else:
                    closure()
                
                #save losses
                if self.neglect_p_dyn==False: 
                    phys_loss_p_sum+=0#TODO
                    phys_loss_p_sum_scaled+=0#*self.loss_lam_final[2]#TODO
                if self.n_collo>0: 
                    phys_loss_qd_sum+=self.closure_losses[1]
                    phys_loss_qd_sum_scaled+=self.closure_losses[1]*self.loss_lam_final[1]
                    if self.first_order_dyn==False:
                        phys_loss_qdd_sum+=self.closure_losses[3]
                        phys_loss_qdd_sum_scaled+=self.closure_losses[3]*self.loss_lam_final[3]
                if self.n_data>0: 
                    data_loss_sum+=self.closure_losses[2]
                    data_loss_sum_scaled+=self.closure_losses[2]*self.loss_lam_final[2]
                if self.n_ic>0: 
                    ic_loss_sum+=self.closure_losses[0]
                    ic_loss_sum_scaled+=self.closure_losses[0]*self.loss_lam_final[0]
                loss_sum += self.closure_losses[-1]
                
                
            loss_mean = loss_sum / len(dataloader)
            loss_p_mean = phys_loss_p_sum/ len(dataloader)
            loss_p_mean_scaled = phys_loss_p_sum_scaled/ len(dataloader)
            loss_qd_mean = phys_loss_qd_sum/ len(dataloader)
            loss_qd_mean_scaled = phys_loss_qd_sum_scaled/ len(dataloader)
            loss_qdd_mean = phys_loss_qdd_sum/ len(dataloader)
            loss_qdd_mean_scaled = phys_loss_qdd_sum_scaled/ len(dataloader)
            loss_data_mean = data_loss_sum/ len(dataloader)
            loss_data_mean_scaled = data_loss_sum_scaled/ len(dataloader)
            loss_ic_mean = ic_loss_sum/ len(dataloader)
            loss_ic_mean_scaled = ic_loss_sum_scaled/ len(dataloader)
            
            self.saveLosses(epoch,train_mode,loss_p_mean,loss_p_mean_scaled,loss_qd_mean,loss_qd_mean_scaled,
                            loss_qdd_mean,loss_qdd_mean_scaled,
                           loss_data_mean,loss_data_mean_scaled,loss_ic_mean,loss_ic_mean_scaled,loss_mean)
            
            #loss weights
            if train_mode and self.pinn_flag and epoch%self.loss_weighting_interval==0:
                if self.loss_weighting=="naive" and epoch==0 and self.retrain_network[0]==False:
                    self.naiveInitLossLam(loss_ic_mean,loss_qd_mean,loss_data_mean,loss_qdd_mean,False)                    
                if self.loss_weighting=="softadapt":
                    if epoch==0 and self.retrain_network[0]==False:
                        if self.softAdapt_naive_flag:
                            self.naiveInitLossLam(loss_ic_mean,loss_qd_mean,loss_data_mean,loss_qdd_mean,False)
                        self.softAdapt_naive_lam=copy.copy(self.loss_lam_final)
                    if self.n_data>0:
                        current_losses=np.multiply(np.array((loss_ic_mean,loss_qd_mean,loss_data_mean)),self.softAdapt_naive_lam[:-1])
                    else:
                        current_losses=np.multiply(np.array((loss_ic_mean,loss_qd_mean)),self.softAdapt_naive_lam[:-2])
                    if epoch>0:
                        self.softAdapt(current_losses)
                    self.softAdapt_old_losses=copy.copy(current_losses)
                
    def softAdapt(self,current_losses):
        #approach from Heydari2019 with adaptation
        #https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/theory/advanced_schemes.html#softadapt
        epsilon=1e-8
        n=len(current_losses)
        exp_args=np.zeros([n])
        counter=np.zeros([n])
        for i in range(n):
            exp_args[i]=current_losses[i]/(self.softAdapt_old_losses[i]+epsilon)
        mu=np.max(exp_args)
        for i in range(n):
            exp_args[i]-=mu
            counter[i]=math.exp(exp_args[i])
        sum_softmax=np.sum(counter)
        for i in range(n):
            temp_lam_new=self.softAdapt_naive_lam[i]*counter[i]/sum_softmax
            self.loss_lam_final[i]=(1-self.alpha_weighting)*self.loss_lam_final[i] + self.alpha_weighting*temp_lam_new
    
    def LRA(self,data_loss,ic_loss,phys_loss):
        # learning rate annealing from Wang2021
        grads_data, grads_ic, grads_phys = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                raise ValueError('Untrainable net parameters detected during loss weighting!')
            if self.n_data >0:
                grad_data = torch.autograd.grad(data_loss, p, retain_graph=True)[0].detach().reshape(-1,1)
                grads_data.append(copy.copy(grad_data))
            if self.n_ic>0:
                grad_ic = torch.autograd.grad(ic_loss, p, retain_graph=True)[0].detach().reshape(-1,1)
                grads_ic.append(copy.copy(grad_ic))
            #in the last layer, the bias terms have no influence on the phys loss, as they are eliminated by the derivation
            if n!='layer_out.bias':
                grad_phys = torch.autograd.grad(phys_loss, p, retain_graph=True)[0].detach().reshape(-1,1)
                grads_phys.append(copy.copy(grad_phys))
        
        # stack overall gradient of net parameters with respect to individual loss terms
        grad_loss_phys = torch.vstack(tuple(grads_phys)).detach().numpy()
        if len(grads_ic)>0:
            grad_loss_ic = torch.vstack(tuple(grads_ic)).detach().numpy()
            lam_ic_new = abs(grad_loss_phys).max() / abs(grad_loss_ic).mean()
        else:
            lam_ic_new=1 
        if len(grads_data)>0: 
            grad_loss_data = torch.vstack(tuple(grads_data)).detach().numpy()
            lam_data_new = abs(grad_loss_phys).max() / abs(grad_loss_data).mean()
        else: 
            lam_data_new=1 
          
        # update the weights with moving average
        self.loss_lam_final[2] = (1-self.alpha_weighting)*self.loss_lam_final[2] + self.alpha_weighting*lam_data_new
        self.loss_lam_final[0] = (1-self.alpha_weighting)*self.loss_lam_final[0] + self.alpha_weighting*lam_ic_new
        
    def saveLosses(self,epoch,train_mode,loss_p_mean,loss_p_mean_scaled,loss_qd_mean,loss_qd_mean_scaled,loss_qdd_mean,loss_qdd_mean_scaled,
                   loss_data_mean,loss_data_mean_scaled,loss_ic_mean,loss_ic_mean_scaled,loss_mean):
        if train_mode:
            if loss_p_mean>0: self.train_loss_history["phys_p"][epoch]=loss_p_mean
            if loss_qd_mean>0: self.train_loss_history["phys_qd"][epoch]=loss_qd_mean
            if loss_qdd_mean>0: self.train_loss_history["phys_qdd"][epoch]=loss_qdd_mean
            if loss_data_mean>0: self.train_loss_history["data"][epoch]=loss_data_mean
            if loss_ic_mean>0: self.train_loss_history["ic"][epoch]=loss_ic_mean
            if loss_mean>0: self.train_loss_history["total"][epoch]=loss_mean
            if loss_p_mean_scaled>0: self.train_loss_history["phys_p_scaled"][epoch]=loss_p_mean_scaled
            if loss_qd_mean_scaled>0: self.train_loss_history["phys_qd_scaled"][epoch]=loss_qd_mean_scaled
            if loss_qdd_mean_scaled>0: self.train_loss_history["phys_qdd_scaled"][epoch]=loss_qdd_mean_scaled
            if loss_data_mean_scaled>0: self.train_loss_history["data_scaled"][epoch]=loss_data_mean_scaled
            if loss_ic_mean_scaled>0: self.train_loss_history["ic_scaled"][epoch]=loss_ic_mean_scaled        
        else:
            if loss_p_mean>0: self.val_loss_history["phys_p"][epoch]=loss_p_mean
            if loss_qd_mean>0: self.val_loss_history["phys_qd"][epoch]=loss_qd_mean
            if loss_qdd_mean>0: self.val_loss_history["phys_qdd"][epoch]=loss_qdd_mean
            if loss_data_mean>0: self.val_loss_history["data"][epoch]=loss_data_mean
            if loss_ic_mean>0: self.val_loss_history["ic"][epoch]=loss_ic_mean
            if loss_mean>0: self.val_loss_history["total"][epoch]=loss_mean
            if loss_p_mean_scaled>0: self.val_loss_history["phys_p_scaled"][epoch]=loss_p_mean_scaled
            if loss_qd_mean_scaled>0: self.val_loss_history["phys_qd_scaled"][epoch]=loss_qd_mean_scaled
            if loss_qdd_mean_scaled>0: self.val_loss_history["phys_qdd_scaled"][epoch]=loss_qdd_mean_scaled
            if loss_data_mean_scaled>0: self.val_loss_history["data_scaled"][epoch]=loss_data_mean_scaled
            if loss_ic_mean_scaled>0: self.val_loss_history["ic_scaled"][epoch]=loss_ic_mean_scaled
    
    def plotPerformance(self,foldername,epoch):
        #save params
        if self.pinn_flag:
            name="pinn_downsampled_"+str(self.factor_downsampling)
        else:
            name="rnn_downsampled_"+str(self.factor_downsampling)
        nn_params={"n_neurons":self.n_neurons,
                   "lr_init":self.lr_init,
                   "n_hidden":self.n_hidden,
                   "input_dim":self.input_dim,
                   "output_dim":self.output_dim,
                   "dropout_GRU":self.dropout_GRU,
                   "factor_downsampling":self.factor_downsampling,
                   "act_str":self.act_str,
                   "betamax":self.betamax,
                   "betamin":self.betamin,
                   "mEmax":self.mEmax,
                   "mEmin":self.mEmin,
                   "pmax":self.pmax,
                   "qmax":self.qmax,
                   "qdmax":self.qdmax,
                   "m_x":self.m_x,
                   "b_x":self.b_x,
                   "n_akt":self.n_akt,
                   "udim":self.udim,
                   "xdim":self.xdim,
                   "t_sample":self.t_sample,
                   "T":self.T,
                   "ddpinn_flag": self.ddpinn_flag,
                   "ddpinn_props": self.ddpinn_props,
                   "blackbox_mode": self.blackbox_mode}
        params_file=open(Path(foldername,name+"_params"),'wb') 
        pickle.dump(nn_params,params_file)
        params_file.close()
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), Path(foldername,name+".pt"))
        
        if self.pinn_flag:
            #export json for C++ for use at test bench
            weights_dic={"w0":self.model.layer_in.weight.detach().numpy(),
                         "w"+str(self.n_hidden+1):self.model.layer_out.weight.detach().numpy()}
            biases_dic={"b0":self.model.layer_in.bias.detach().numpy()[:,np.newaxis],
                        "b"+str(self.n_hidden+1):self.model.layer_out.bias.detach().numpy()[:,np.newaxis]}
            for i in range(self.n_hidden):  
                weights_dic["w"+str(i+1)]=self.model.layers_middle[i].weight.detach().numpy()
                biases_dic["b"+str(i+1)]=self.model.layers_middle[i].bias.detach().numpy()[:,np.newaxis]
            weights_serializable = {key: value.tolist() for key, value in weights_dic.items()}
            biases_serializable = {key: value.tolist() for key, value in biases_dic.items()}
            nn_params["weights"]=weights_serializable
            nn_params["biases"]=biases_serializable
            with open(Path(foldername,name+".json"), 'w') as file:
                json.dump(nn_params, file,indent=4)
        
        if epoch==self.num_epochs-1:
            #save one object for retraining at the end of the training
            #remove sampled data to save memory
            self.train_DL_Pinn=None
            self.val_DL_Pinn=None
            file=open(Path(foldername,"pinn_obj"),'wb') 
            pickle.dump(self,file)
            file.close()
        if self.hpo_flag==False:# or epoch==self.num_epochs-1:
            self.model.eval() 
            torch.set_grad_enabled(False)
            #Test PINN on whole dataset for training/validation
            beta_train_scaled=normalizeVar(self.beta_train_raw, self.m_x["beta"], self.b_x["beta"], True)
            mE_train_scaled=normalizeVar(self.mE_train_raw, self.m_x["mE"], self.b_x["mE"], True)
            pinn_pred_raw=np.zeros([self.X_raw_train.shape[0],4*self.n_akt])
            #self-loop prediction
            h_step=torch.zeros(self.model.n_hidden, 1, self.model.n_neurons)
            X_scaled_train=normalizeX(self.X_raw_train,self.neglect_p_dyn,self.first_order_dyn,self.indices,self.m_x,
                                      self.b_x,self.coldim,True,self.xdim,self.udim,False)
            x_k_torch_scaled=X_scaled_train[0,:self.xdim]
            U_scaled_train=normalizeVar(self.U_raw_train,self.m_x["p"],self.b_x["p"],True)
            for i in range(U_scaled_train.shape[0]):
                u_k_scaled=U_scaled_train[i,:]
                #PINN
                x_k_torch_scaled,h_step,pinn_pred_raw_q,pinn_pred_raw_qd,pinn_pred_raw_p=predictOneStep(self.model,h_step,
                                                 x_k_torch_scaled,u_k_scaled,self.xdim,self.udim,beta_train_scaled,mE_train_scaled,
                                                 self.coldim,self.t_sample_scaled,self.pinn_flag,
                                                 self.n_akt,self.m_x,self.b_x,self.neglect_p_dyn,self.first_order_dyn, self.blackbox_mode)
                pinn_pred_raw[i,0:self.n_akt]=pinn_pred_raw_q
                if self.first_order_dyn==False:
                    pinn_pred_raw[i,self.n_akt:2*self.n_akt]=pinn_pred_raw_qd
                if self.neglect_p_dyn==False: 
                    pinn_pred_raw[i,self.n_akt:]=pinn_pred_raw_p
            #Plot
            time = np.arange(U_scaled_train.shape[0])*self.t_sample
            beta_lbl=str(round(self.beta_train_raw*180/math.pi))
            mE_lbl_grams=str(round(self.mE_train_raw*1000))
            groundtruth=self.Y_raw_train
            if self.first_order_dyn:
                groundtruth=np.hstack([groundtruth, np.zeros([groundtruth.shape[0],self.n_akt])])          
            if self.neglect_p_dyn:
                groundtruth=np.hstack([groundtruth, np.zeros([groundtruth.shape[0],2*self.n_akt])])          
            for i in range(self.n_akt):
                i_act=str(i+1)
                y_data=np.transpose([groundtruth[:,2*self.n_akt+i*2]*10**(-5),pinn_pred_raw[:,2*self.n_akt+i*2]*10**(-5),
                                                     groundtruth[:,2*self.n_akt+1+i*2]*10**(-5),pinn_pred_raw[:,2*self.n_akt+1+i*2]*10**(-5),
                                                     groundtruth[:,i]*180/math.pi,pinn_pred_raw[:,i]*180/math.pi,
                                                     groundtruth[:,self.n_akt+i]*180/math.pi,pinn_pred_raw[:,self.n_akt+i]*180/math.pi])
                y_labels_notex=["p_{"+i_act+",1} in bar",
                 "p_{"+i_act+",2} in bar","q_"+i_act+" in deg","qd_"+i_act+" in deg/s"]
                pinnphys.SimplePlot((18, 16), time, y_data,["Train/Val data","NN"],["-","--"], None, "time in s", 
                                   y_labels_notex,Path(foldername,"act"+i_act+"_beta"+beta_lbl+"_mE"+mE_lbl_grams+"_train."+self.img_format),4)
                
            #Test PINN vs. ODE on test trajectory (only first-order dynamics to speed up the test!)
            if self.mpc_horizon>0: 
                mpc_flags=[True, False]
            else: 
                mpc_flags=[False]
            for plot_mpc in mpc_flags:
                if plot_mpc: 
                    mpc_label="_mpc"
                else: 
                    mpc_label=""
                for beta_test_raw in pinnphys.deg2rad(np.linspace(0,90,self.plot_stages)):
                    for mE_test_raw in np.linspace(0,0.2,self.plot_stages):
                        beta_test_scaled=normalizeVar(beta_test_raw, self.m_x["beta"], self.b_x["beta"], True)
                        mE_test_scaled=normalizeVar(mE_test_raw, self.m_x["mE"], self.b_x["mE"], True)
                        pinn_pred_raw=np.zeros([self.U_raw_test.shape[0],4*self.n_akt])
                        maple_pred_raw=np.zeros(pinn_pred_raw.shape)
                        #self-loop prediction
                        h_step=torch.zeros(self.model.n_hidden, 1, self.model.n_neurons)
                        X_scaled_test=normalizeX(self.X_raw_test,self.neglect_p_dyn,self.first_order_dyn,self.indices,self.m_x,
                                                  self.b_x,self.coldim,True,self.xdim,self.udim,False)
                        x_k_torch_scaled=X_scaled_test[0,:self.xdim]
                        if self.first_order_dyn:
                            x_k_maple_raw=self.X_raw_test[0,:3*self.n_akt]#q,p1,p2
                        else:
                            x_k_maple_raw=self.X_raw_test[0,:4*self.n_akt]#q,qd,p1,p2
                            x_k_maple_raw=np.delete(x_k_maple_raw,np.arange(self.n_akt)+self.n_akt,0)
                        g_test_raw=pinnphys.calc_g_KS0(beta_test_raw, self.g_KSW.detach().numpy(),self.rotate_y)
                        qd_old_red=np.zeros(self.n_akt)
                        for i in range(self.U_raw_test.shape[0]):
                            u_k_raw=self.U_raw_test[i,:]
                            u_k_scaled=self.U_scaled_test[i,:]
                            if i%self.factor_downsampling==0:
                                x_k_torch_scaled,h_step,pinn_pred_raw_q,pinn_pred_raw_qd,pinn_pred_raw_p=predictOneStep(self.model,
                                                         h_step,x_k_torch_scaled,u_k_scaled,self.xdim,self.udim,beta_test_scaled,
                                                         mE_test_scaled,self.coldim,self.t_sample_scaled,self.pinn_flag,
                                                         self.n_akt,self.m_x,
                                                         self.b_x,self.neglect_p_dyn,self.first_order_dyn,self.blackbox_mode)
                                pinn_pred_raw[i,0:self.n_akt]=pinn_pred_raw_q
                                if self.first_order_dyn==False:
                                    pinn_pred_raw[i,self.n_akt:2*self.n_akt]=pinn_pred_raw_qd
                                if self.neglect_p_dyn==False: 
                                    pinn_pred_raw[i,2*self.n_akt:]=pinn_pred_raw_p
                            #ODE
                            x_k_maple_raw,qd_old_red=pinnphys.sim_dyn_nl_red(x_k_maple_raw,u_k_raw,1/self.f_sample_data,self.n_akt,mE_test_raw,
                                                                 g_test_raw,True,self.neglect_p_dyn,with_rc=self.with_rc,qd_old=qd_old_red,qd_c=self.qd_c)
                            maple_pred_raw[i,:self.n_akt]=x_k_maple_raw[0,:self.n_akt]
                            maple_pred_raw[i,2*self.n_akt:]=x_k_maple_raw[0,self.n_akt:]
                            if plot_mpc and i%self.mpc_horizon==0 and i>0:
                                #MPC simulation
                                x_k_torch_scaled[:,:self.xdim]=x_k_maple_raw[:,:self.xdim]
                                x_k_torch_scaled[:,:self.n_akt]=normalizeVar(x_k_torch_scaled[:,:self.n_akt], self.m_x["q"], self.b_x["q"], True)
                                if self.neglect_p_dyn==False:
                                    x_k_torch_scaled[:,self.n_akt:]=normalizeVar(x_k_torch_scaled[:,self.n_akt:], self.m_x["p"], self.b_x["p"], True)
                                
                        #Plot
                        beta_lbl=str(round(beta_test_raw*180/math.pi))
                        mE_lbl_grams=str(round(mE_test_raw*1000))
                        plot_idx = np.arange(0, self.U_raw_test.shape[0], self.factor_downsampling)
                        time=plot_idx* 1/self.f_sample_data
                        for i in range(self.n_akt):
                            i_act=str(i+1)
                            y_data=np.transpose([maple_pred_raw[plot_idx,2*self.n_akt+i*2]*10**(-5),pinn_pred_raw[plot_idx,2*self.n_akt+i*2]*10**(-5),
                                             maple_pred_raw[plot_idx,2*self.n_akt+1+i*2]*10**(-5),pinn_pred_raw[plot_idx,2*self.n_akt+1+i*2]*10**(-5),
                                             maple_pred_raw[plot_idx,i]*180/math.pi,pinn_pred_raw[plot_idx,i]*180/math.pi,
                                             maple_pred_raw[plot_idx,self.n_akt+i]*180/math.pi,pinn_pred_raw[plot_idx,self.n_akt+i]*180/math.pi])
                            y_labels=[r"$p_{"+i_act+",1}$ in bar",
                             r"$p_{"+i_act+",2}$ in bar",r"$q_"+i_act+"$ in deg",r"$qd_"+i_act+"$ in deg/s"]
                            y_labels_notex=["p_{"+i_act+",1} in bar",
                             "p_{"+i_act+",2} in bar","q_"+i_act+" in deg","qd_"+i_act+" in deg/s"]
                            nSubplots=4
                            pinnphys.SimplePlot((18, 16), time, y_data,["ODE","NN"],["-","--"], None, "time in s", 
                                               y_labels_notex,Path(foldername,"act"+i_act+"_beta"+beta_lbl+"_mE"+mE_lbl_grams+mpc_label+"_test."+self.img_format),nSubplots)
                    
        #Plot losses
        if epoch>0:
            y_labels_loss_notex=["L","L_ic","L_d","L_phys_qd","L_phys_qdd"]
            y_labels_loss_scaled_notex=["L","l_ic*L_ic","l_d*L_d","l_phys_qd*L_phys_qd","l_phys_qdd*L_phys_qdd"]
            plot_loss=np.transpose(np.array([self.train_loss_history["total"][:epoch+1],self.val_loss_history["total"][:epoch+1],
                              self.train_loss_history["ic"][:epoch+1],self.val_loss_history["ic"][:epoch+1],
                              self.train_loss_history["data"][:epoch+1],self.val_loss_history["data"][:epoch+1],
                              self.train_loss_history["phys_qd"][:epoch+1],self.val_loss_history["phys_qd"][:epoch+1],
                              self.train_loss_history["phys_qdd"][:epoch+1],self.val_loss_history["phys_qdd"][:epoch+1]]))
            plot_loss_scaled=np.transpose(np.array([self.train_loss_history["total"][:epoch+1],self.val_loss_history["total"][:epoch+1],
                              self.train_loss_history["ic_scaled"][:epoch+1],self.val_loss_history["ic_scaled"][:epoch+1],
                              self.train_loss_history["data_scaled"][:epoch+1],self.val_loss_history["data_scaled"][:epoch+1],
                              self.train_loss_history["phys_qd_scaled"][:epoch+1],self.val_loss_history["phys_qd_scaled"][:epoch+1],
                              self.train_loss_history["phys_qdd_scaled"][:epoch+1],self.val_loss_history["phys_qdd_scaled"][:epoch+1]]))
            plot_lam=np.transpose(np.array([self.loss_lam_history[:epoch+1,0],self.loss_lam_history[:epoch+1,0],
                              self.loss_lam_history[:epoch+1,1],self.loss_lam_history[:epoch+1,1],
                              self.loss_lam_history[:epoch+1,2],self.loss_lam_history[:epoch+1,2],
                              self.loss_lam_history[:epoch+1,3],self.loss_lam_history[:epoch+1,3]]))
            y_labels_lam_notex=["l_ic","l_phys_qd","l_d","l_phys_qdd"]
            
            pinnphys.SimplePlot((18, 16), np.arange(epoch+1),self.learning_rate_history[:epoch+1],["-","-"],["-","--"], None, "epoch","learning rate", 
                                                str(foldername) +"/learning_rate."+self.img_format,1)
            pinnphys.SimplePlot((18, 16), np.arange(epoch+1),plot_lam,["-","-"],["-","--"], None, "epoch",y_labels_lam_notex, 
                                                str(foldername) +"/lam_loss."+self.img_format,len(y_labels_lam_notex))
            for start_i in [0,epoch-49]:
                if start_i==0:
                    temp_str="global"
                elif start_i<0:
                    break
                else:
                    temp_str="local"
                pinnphys.SimplePlot((18, 16), np.arange(epoch+1)[start_i:], plot_loss[start_i:,:],["Training","Validation"],["-","--"], None, "epoch",y_labels_loss_notex, 
                                                    str(foldername) +"/losses_"+temp_str+"."+self.img_format,len(y_labels_loss_notex),None,False,True)
                pinnphys.SimplePlot((18, 16), np.arange(epoch+1)[start_i:],plot_loss_scaled[start_i:,:],["Training","Validation"],["-","--"], None, "epoch",y_labels_loss_scaled_notex, 
                                                str(foldername) +"/losses_scaled_"+temp_str+"."+self.img_format,len(y_labels_loss_scaled_notex),None,False,True)
            
            #export stats
            stats={"train_loss_history": self.train_loss_history,
                   "val_loss_history": self.val_loss_history,
                   "loss_lam_history": self.loss_lam_history,
                   "learning_rate_history": self.learning_rate_history,
                   "epoch_time_history_s":self.epoch_time_history_s,
                   "epoch_idx": epoch+1}
            stats_file=open(Path(foldername,"stats"),'wb') 
            pickle.dump(stats,stats_file)
            stats_file.close()
              
    def sampleColloInit(self,red_output_dim):
        sampler = qmc.LatinHypercube(d=self.input_dim)
        collo_points = sampler.random(n=self.n_collo)
        l_bounds = -np.ones(self.input_dim)
        u_bounds = np.ones(self.input_dim)
        collo_points=qmc.scale(collo_points, l_bounds, u_bounds)
        #generate multivariate normal samples with mean at 0 (only for xdim states)
        if self.sample_normal_dist_x:
            mean = np.zeros(self.xdim)
            std_deviation=0.4
            covariance = np.eye(self.xdim) * std_deviation**2 
            x_collo_points = np.random.multivariate_normal(mean, covariance, self.n_collo)
            collo_points[:,:self.xdim]=x_collo_points
        #create points from collocation points for initial conditions
        ic_points=copy.copy(collo_points)
        np.random.shuffle(ic_points)
        ic_points=ic_points[:self.n_ic,:]
        ic_points[:,-1]=-1#scaled t=0, x(t=0)=x_0 must be considered later in outputs
        #merge collocation, ic and data points randomly
        data_collo_merged=np.zeros([self.n_data+self.n_collo+self.n_ic,self.input_dim+red_output_dim+1])#last col as flag for data or collo
        if self.n_data>0:
            data_collo_merged[:self.n_data,:-1]=self.data_points
            data_collo_merged[:self.n_data,-1]=2#data point -> last col 2
        if self.n_ic>0:
            data_collo_merged[self.n_data:self.n_data+self.n_ic,:-self.xdim-1]=ic_points
            data_collo_merged[self.n_data:self.n_data+self.n_ic,-self.xdim-1:-1]=ic_points[:,:self.xdim]#x(t=0)=x_0 
            data_collo_merged[self.n_data:self.n_data+self.n_ic,-1]=3#ic point -> last col 3
        data_collo_merged[self.n_data+self.n_ic:,:-self.xdim-1]=collo_points
        data_collo_merged[self.n_data+self.n_ic:,-1]=1#collocation point -> last col 1
        #shuffle
        np.random.shuffle(data_collo_merged)
        #split
        train_data,val_data=self.splitData(data_collo_merged,self.n_data+self.n_collo+self.n_ic)
        #batches    
        train_data=TorchDataset(train_data[:,:self.input_dim],train_data[:,self.input_dim:],self.input_dim, self.batch_size)
        val_data=TorchDataset(val_data[:,:self.input_dim],val_data[:,self.input_dim:],self.input_dim, self.batch_size)
        #dataLoader
        train_DL = torch.utils.data.DataLoader(train_data,batch_size = 1,shuffle = False)
        val_DL = torch.utils.data.DataLoader(val_data,batch_size = 1,shuffle = False)
        self.train_DL_Pinn=train_DL
        self.val_DL_Pinn=val_DL
        
    
    def naiveInitLossLam(self,ic_loss,phys_loss_qd,data_loss,phys_loss_qdd,detach=True):
        #naive initialization of loss_lam in first iteration
        loss_lam=np.zeros([self.n_losses])
        if detach:
            ic_loss=ic_loss.detach().numpy()
            phys_loss_qd=phys_loss_qd.detach().numpy()
            phys_loss_qdd=phys_loss_qdd.detach().numpy()
            if self.n_data>0:
                data_loss=data_loss.detach().numpy()
            # if self.neglect_p_dyn==False:
            #     phys_loss_p=phys_loss_p.detach().numpy()  
        if self.first_order_dyn:
            temp=np.array([ic_loss,phys_loss_qd,data_loss])
        else:
            temp=np.array([ic_loss,phys_loss_qd,data_loss,phys_loss_qdd])
        max_loss=np.max(temp)
        for i in range(temp.shape[0]):
            if temp[i]>0:
                loss_lam[i]=max_loss/temp[i]
            else:
                loss_lam[i]=1
        self.loss_lam_final=loss_lam
    
    def addColVar(self):
        #colvar and t
        if self.coldim==1:
            colvar_train_raw=np.array([self.beta_train_raw])
        elif self.coldim==2:
            colvar_train_raw=np.array([self.beta_train_raw,self.mE_train_raw])
        X_raw_train_added=np.hstack([self.X_raw_train,np.ones([self.n_data,self.coldim])*colvar_train_raw,
                                np.ones([self.n_data,1])*self.t_sample])
        return np.hstack([X_raw_train_added,self.Y_raw_train])
    
    def splitData(self,data,n_data_total):
        #split in train und val
        if self.batch_size>0:
            split_point = int(self.split_size * n_data_total)-int(self.split_size * n_data_total)%self.batch_size
        else:
            split_point = int(self.split_size * n_data_total)
        train_data = data[:split_point,:]
        val_data = data[split_point:,:]
        if self.batch_size>0: 
            n_cut=len(val_data)%self.batch_size
            if n_cut>0:
                val_data = data[split_point:-n_cut,:]
        return train_data,val_data
    
    def noPinnDL(self):
        n_dp=self.data_points.shape[0]
        data=np.zeros([n_dp,self.input_dim+self.output_dim+1])#last col as flag for data
        data[:,:-1]=self.data_points
        #delete colvar and t
        data=np.delete(data,np.arange(self.xdim+self.udim,self.xdim+self.udim+self.coldim+1),1)
        data[:,-1]=2#data point -> last col 2
        #split
        train_data,val_data=self.splitData(data,n_dp)
        #batches
        self.input_dim=self.input_dim-self.coldim-1
        train_data=TorchDataset(train_data[:,:self.input_dim],train_data[:,self.input_dim:],self.input_dim, self.batch_size)
        val_data=TorchDataset(val_data[:,:self.input_dim],val_data[:,self.input_dim:],self.input_dim, self.batch_size)
        #dataLoader
        self.train_DL_noPinn = torch.utils.data.DataLoader(train_data,batch_size = 1,shuffle = False)
        self.val_DL_noPinn = torch.utils.data.DataLoader(val_data,batch_size = 1,shuffle = False)
          
def loadData(pfad_data,cut_sample,n_akt,xdim,udim,first_order_dyn,neglect_p_dyn,factor_downsampling=1):
    data_raw = read_csv(pfad_data, sep = ',', decimal = '.')
    data_raw = data_raw.to_numpy()
    #possible downsampling via factor_downsampling
    data_raw=data_raw[np.arange(0,data_raw.shape[0],factor_downsampling),:]
    if cut_sample>0 and cut_sample<data_raw.shape[0]:
        data_raw=data_raw[:cut_sample+1,:]
    beta_data=data_raw[0,1]
    mE_data=data_raw[0,2]
    t_sample_true=data_raw[1,0]-data_raw[0,0]
    data_raw=np.delete(data_raw,[0,1,2],1)#time,beta,mE weg
    #data_raw: q1,q2,..,qd1,qd2,..,p11,p12,p21,p22,..,,p11d,p12d,p21d,p22d,..
    q_indices=np.arange(0,n_akt)
    if first_order_dyn:
        #no qd
        indqd=np.arange(n_akt)+n_akt
        data_raw=np.delete(data_raw,indqd,1)
        temp_ind=n_akt
        qd_indices=None
    else:
        temp_ind=2*n_akt
        qd_indices=np.arange(n_akt)+n_akt
    if neglect_p_dyn:
        #no p
        indp=np.arange(2*n_akt)+temp_ind
        data_raw=np.delete(data_raw,indp,1)
        p_indices=None
    else:
        p_indices=np.arange(2*n_akt)+temp_ind
    x_indices=np.arange(0,xdim)
    u_indices=np.arange(xdim,xdim+udim)
    indices={"x":x_indices,"u":u_indices,"q":q_indices,"qd":qd_indices,"p":p_indices}#pd_indices=u_indices
    X_raw=data_raw[:-1,:]
    Y_raw=data_raw[1:,:]
    Y_raw=Y_raw[:,indices["x"]]
    return X_raw,Y_raw,indices,beta_data,mE_data,t_sample_true

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, input_data,output_data,input_dim, batch_size):
        self.batch_size=batch_size
        datensatz_pytorch = ()
        tensor_inputs = torch.from_numpy(input_data).type(torch.get_default_dtype())
        tensor_outputs = torch.from_numpy(output_data).type(torch.get_default_dtype())
        tensor_ges = torch.hstack((tensor_inputs,tensor_outputs))
        if self.batch_size>0:
            self.datensatz_pytorch = datensatz_pytorch + torch.split(tensor_ges, self.batch_size)
        else:
            #full batch
            self.datensatz_pytorch = tensor_ges
        self.input_dim=input_dim
    def __len__(self):
        if self.batch_size>0:
            return len(self.datensatz_pytorch)
        else:
            return 1
    def __getitem__(self,idx):
        if self.batch_size>0:
            daten = self.datensatz_pytorch[idx]
        else:
            daten=self.datensatz_pytorch
        X_daten = daten[:,:self.input_dim]
        Y_daten = daten[:,self.input_dim:]
        return X_daten, Y_daten
      
class FNN(torch.nn.Module):
    def __init__(self,input_dim, n_hidden, n_neurons, output_dim, act_str,ddpinn_flag,ddpinn_props):
        super().__init__()
        print("Init FNN with "+act_str+"-activation, "+str(n_hidden)+" hidden layers and "+str(n_neurons)+" neurons per layer\n")
        self.act_str=act_str
        self.init_mode='xavier_uniform'
        self.ddpinn_flag=ddpinn_flag
        self.input_dim=input_dim
        self.output_dim=output_dim
        if self.ddpinn_flag:
            self.ansatz=ddpinn_props["ansatz"]
            self.n_ansatz=ddpinn_props["n_ansatz"]
            if self.ansatz=="damped":
                self.n_per_ansatz=4
            self.n_params_ansatz=self.n_per_ansatz*self.n_ansatz*self.output_dim#number of predicted params for all ansatz functions
            self.layer_in = torch.nn.Linear(input_dim-1, n_neurons)#t decoupled
            self.layer_out = torch.nn.Linear(n_neurons,self.n_params_ansatz)#predict ansatz coefficients
            self.act_g=torch.sin
            self.act_g_derivative=torch.cos
        else:
            self.layer_in = torch.nn.Linear(input_dim, n_neurons)
            self.layer_out = torch.nn.Linear(n_neurons,output_dim)
        
        if self.act_str=="Tanh":
            self.act=torch.nn.Tanh()
        elif self.act_str=="Sigmoid":
            self.act=torch.nn.Sigmoid()
        elif self.act_str=="SELU":
            self.act=torch.nn.SELU()
        elif self.act_str=="GELU":
            self.act=torch.nn.GELU()
        else:
            raise ValueError("Activation function not supported")
        self.n_neurons = n_neurons
        layers=[]
        for i in range(n_hidden):
            layers.append(torch.nn.Linear(n_neurons,n_neurons))
        
        self.layers_middle=torch.nn.ModuleList(layers)
        self.n_hidden=n_hidden
        #init weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            # weights:
            if self.init_mode == 'xavier_uniform':
                try: gain = torch.nn.init.calculate_gain(self.act_str.lower())
                except: gain=1.0
                torch.nn.init.xavier_uniform_(module.weight, gain)  # only applies to nn.Linear
            # bias:
            if module.bias is not None:
                module.bias.data.zero_()        
        return None

    def forward(self, X,derivative_flag=False):
        if self.ddpinn_flag:
            t=X[0,0,-1]+1#t_raw=0s equals t_scaled=-1 
            X=X[0,0,:-1].view(1,1,self.input_dim-1)
                    
        out = self.act(self.layer_in(X))
        for layer in self.layers_middle:
            out=self.act(layer(out))
        out=self.layer_out(out)
        
        if self.ddpinn_flag:
            a_full=out.view(self.n_per_ansatz,self.n_ansatz*self.output_dim)
            if self.ansatz=='damped' and self.n_per_ansatz==4:
                a,b,c,d=a_full[0],a_full[1],a_full[2],a_full[3]
                bt_plus_c=b*t+c
                exp_neg_dt=torch.exp(-d*t)
                act_g_bt_plus_c=self.act_g(bt_plus_c)
                
                g_out=a*(act_g_bt_plus_c*exp_neg_dt-self.act_g(c))
                if derivative_flag: dgdt_out=a*(b*self.act_g_derivative(bt_plus_c)-d*act_g_bt_plus_c)*exp_neg_dt
            else:
                raise ValueError("DD-PINN properties not supported!")
            g_out_sum=g_out.view(self.output_dim, self.n_ansatz).sum(dim=1)
            out=(X[0,0,:self.output_dim]+g_out_sum).view(1,1,self.output_dim)
            if derivative_flag:
                doutdt=dgdt_out.view(self.output_dim, self.n_ansatz).sum(dim=1)
                out=(out,doutdt.view(1,1,self.output_dim))
        return out
    
class GRU(torch.nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons, output_dim,dropout_GRU):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.dropout_GRU=dropout_GRU
        self.GRU = torch.nn.GRU(input_dim, n_neurons, n_hidden, batch_first = True,dropout=dropout_GRU) # batch_first = True bewirkt, dass Input/Output Tensoren der Form batch_size, n_samples, n_neurons ist
        self.linout = torch.nn.Linear(n_neurons, output_dim)
    def forward(self, x, h_t):
        batch_size = 1
        x = x.reshape(batch_size, -1, self.input_dim) 
        out, h_t = self.GRU(x, h_t) 
        out = self.linout(out)
        out.reshape((1,self.output_dim))
        h_t.reshape((self.n_hidden,1,self.n_neurons))
        return out, h_t
    
# Blackbox-FNN with delta_x as output
class FNN_blackbox(torch.nn.Module):
    def __init__(self,input_dim, n_hidden, n_neurons, output_dim, act_str):
        super().__init__()
        self.act_str=act_str
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.layer_in = torch.nn.Linear(input_dim, n_neurons)
        self.layer_out = torch.nn.Linear(n_neurons,output_dim)
        self.n_neurons = n_neurons
        layers=[]
        for i in range(n_hidden):
            layers.append(torch.nn.Linear(n_neurons,n_neurons))
        
        self.layers_middle=torch.nn.ModuleList(layers)
        self.n_hidden=n_hidden
        
        if self.act_str=="Tanh":
            self.act=torch.nn.Tanh()
        elif self.act_str=="Sigmoid":
            self.act=torch.nn.Sigmoid()
        elif self.act_str=="SELU":
            self.act=torch.nn.SELU()
        elif self.act_str=="GELU":
            self.act=torch.nn.GELU()
            
    def forward(self, X):            
        out = self.act(self.layer_in(X))
        for layer in self.layers_middle:
            out=self.act(layer(out))
        out=self.layer_out(out)            
        return out+X[0,0,:self.output_dim]

def retrain(nn_name,cwd,new_optimizer,new_scheduler_min_lr):
    print("Retraining existing network: "+nn_name+"!")
    file = open(nn_name, 'rb')
    try:
        pinn_obj = pickle.load(file)
    except:
        # Solve problems with unix path
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        pinn_obj = pickle.load(file)
    pinn_obj.createFolder(cwd)
    if new_optimizer!=pinn_obj.optimizer_type:
        print("Retraining with different optimizer ("+new_optimizer+", learning rate: "+str(pinn_obj.lr_init)+")!")
        pinn_obj.optimizer_type=new_optimizer
        pinn_obj.optimizer=initOptimizer(new_optimizer,pinn_obj.model.parameters(),pinn_obj.lr_init)
    else:
        print("Retrain with existing optimizer and last learning rate ("+str(pinn_obj.optimizer.param_groups[0]["lr"])+")!")
    if new_optimizer=="lbfgs":
        #only full batch training with lbfgs
        pinn_obj.batch_size=0
    if new_scheduler_min_lr is not None: pinn_obj.scheduler.min_lrs[0]=new_scheduler_min_lr
    pinn_obj.retrain_network=(True, nn_name, new_optimizer,new_scheduler_min_lr)
    pinn_obj.train()
    file.close()

def initOptimizer(optimizer_type,model_parameters,lr_init):
    if optimizer_type=="adam":
        return torch.optim.Adam(model_parameters, lr=lr_init)
    elif optimizer_type=="lbfgs":
        return torch.optim.LBFGS(model_parameters, lr=lr_init,max_iter=10,line_search_fn="strong_wolfe",history_size=10)
    else:
        raise ValueError("Choose lbfgs or adam as optimizer!")
    
def normalizeVar(var,m_var,b_var,flag):
    if flag:
        #normalize
        return var*m_var+b_var
    else:
        #renormalize
        return (var-b_var)*1/m_var  

def normalizeX(data_points_raw,neglect_p_dyn,first_order_dyn,indices,m_x,b_x,coldim,flag,xdim,udim,with_colvar=True):
    data_points=np.zeros(data_points_raw.shape)
    data_points[:,indices["q"]]=normalizeVar(data_points_raw[:,indices["q"]],m_x["q"],b_x["q"],flag)
    if first_order_dyn==False:
        data_points[:,indices["qd"]]=normalizeVar(data_points_raw[:,indices["qd"]],m_x["qd"],b_x["qd"],flag)
    if neglect_p_dyn==False:
        data_points[:,indices["p"]]=normalizeVar(data_points_raw[:,indices["p"]],m_x["p"],b_x["p"],flag)
    data_points[:,indices["u"]]=normalizeVar(data_points_raw[:,indices["u"]],m_x["p"],b_x["p"],flag)
    if with_colvar:
        data_points[:,xdim+udim]=normalizeVar(data_points_raw[:,xdim+udim],m_x["beta"],b_x["beta"],flag)
        if coldim==2:
            data_points[:,xdim+udim+1]=normalizeVar(data_points_raw[:,xdim+udim+1],m_x["mE"],b_x["mE"],flag)
        data_points[:,-1]=normalizeVar(data_points_raw[:,-1],m_x["t"],b_x["t"],flag)
    return data_points

def normalizeY(data_points_raw,neglect_p_dyn,first_order_dyn,indices,m_x,b_x,coldim,flag):
    data_points=np.zeros(data_points_raw.shape)
    data_points[:,indices["q"]]=normalizeVar(data_points_raw[:,indices["q"]],m_x["q"],b_x["q"],flag)
    if first_order_dyn==False:
        data_points[:,indices["qd"]]=normalizeVar(data_points_raw[:,indices["qd"]],m_x["qd"],b_x["qd"],flag)
    if neglect_p_dyn==False:
        data_points[:,indices["p"]]=normalizeVar(data_points_raw[:,indices["p"]],m_x["p"],b_x["p"],flag)
    return data_points

def normalizeXY(data_points_raw,neglect_p_dyn,first_order_dyn,indices,m_x,b_x,coldim,inputdim,flag,xdim,udim):
    data_points=np.zeros(data_points_raw.shape)
    data_points[:,:inputdim]=normalizeX(data_points_raw[:,:inputdim],neglect_p_dyn,first_order_dyn,indices,m_x,b_x,
                                        coldim,flag,xdim,udim)
    data_points[:,inputdim:]=normalizeY(data_points_raw[:,inputdim:],neglect_p_dyn,first_order_dyn,indices,m_x,b_x,coldim,flag)
    return data_points

def predictOneStep(model,h_step,x_k_torch_scaled,u_k_scaled,xdim,udim,beta_scaled,mE_scaled,coldim,t_sample_scaled,pinn_flag,
                   n_akt,m_x,b_x,neglect_p_dyn,first_order_dyn,blackbox_mode):
    pinn_pred_raw_p=None
    pinn_pred_raw_qd=None
    if pinn_flag:
        if coldim==0:
            X_step = torch.hstack((torch.reshape(torch.tensor(x_k_torch_scaled), (1,xdim)), torch.reshape(torch.tensor(u_k_scaled), (1,udim)),
                                   torch.reshape(torch.tensor([t_sample_scaled]),(1,1))))
        else:
            if coldim==1: colvar_tensor=torch.tensor(beta_scaled)
            elif coldim==2: colvar_tensor=torch.tensor((beta_scaled,mE_scaled))
            X_step = torch.hstack((torch.reshape(torch.tensor(x_k_torch_scaled), (1,xdim)), torch.reshape(torch.tensor(u_k_scaled), (1,udim)),
                               torch.reshape(colvar_tensor, (1,coldim)),torch.reshape(torch.tensor([t_sample_scaled]),(1,1))))
        X_step = torch.reshape(X_step, (1, 1, xdim+udim+coldim+1))
        Y_dach = model(X_step)
    else:
        X_step = torch.hstack((torch.reshape(torch.tensor(x_k_torch_scaled), (1,xdim)), torch.reshape(torch.tensor(u_k_scaled), (1,udim))))
        X_step = torch.reshape(X_step, (1, 1, xdim+udim))
        if blackbox_mode=="RNN": Y_dach, h_step = model(X_step, h_step)
        else: Y_dach = model(X_step)
    x_k_torch_scaled=Y_dach[0,0,0:xdim].detach().numpy().reshape([1,xdim])
    pinn_pred_raw_q=normalizeVar(Y_dach[0,0,0:n_akt].detach().numpy(), m_x["q"], b_x["q"], False)
    if first_order_dyn==False:
        pinn_pred_raw_qd=normalizeVar(Y_dach[0,0,n_akt:2*n_akt].detach().numpy(), m_x["qd"], b_x["qd"], False)
    if neglect_p_dyn==False:
        #TODO
        pinn_pred_raw_p=normalizeVar(Y_dach[0,0,n_akt:].detach().numpy(), m_x["p"], b_x["p"], False)  
    return x_k_torch_scaled,h_step,pinn_pred_raw_q,pinn_pred_raw_qd,pinn_pred_raw_p

def RescaleTorchTensor(Y_torch,u_torch,scaler,n_akt):
    q_std=scaler["Y"].scale_[0:n_akt]
    q_mean=scaler["Y"].mean_[0:n_akt]
    p_std=scaler["Y"].scale_[n_akt:]
    p_mean=scaler["Y"].mean_[n_akt:]
    pd_std=scaler["X"].scale_[-2*n_akt:]
    pd_mean=scaler["X"].mean_[-2*n_akt:]
    q_out=torch.zeros((n_akt,1))
    p_out=torch.zeros((2*n_akt,1))
    pd_out=torch.zeros((2*n_akt,1))
    for i in range(n_akt):
        q_out[i,:]=Y_torch[0,0,i]*q_std[i]+q_mean[i]
        p_out[2*i,:]=Y_torch[0,0,n_akt+2*i]*p_std[2*i]+p_mean[2*i]
        p_out[2*i+1,:]=Y_torch[0,0,n_akt+2*i+1]*p_std[2*i+1]+p_mean[2*i+1]
        #p_des
        pd_out[2*i,:]=u_torch[2*i]*pd_std[2*i]+pd_mean[2*i]
        pd_out[2*i+1,:]=u_torch[2*i+1]*pd_std[2*i+1]+pd_mean[2*i+1]     
    return q_out,p_out,pd_out

def start_hpo(params,cwd):
    ray.shutdown()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    ray.init(num_cpus = params.ray_cpus) 
    reporter = tune.CLIReporter(max_progress_rows=100, max_report_frequency=3600)
    def ray_function(config):
        ray_train = ray_hpo('cpu',cwd)
        ray_train.train_fcn(config)
    
    def ray_dir_name(trial):
        name="run_"+str(trial.trial_id)
        return name
    
    def create_storage_path(dir_name):
        current_dir = os.getcwd()
        ray_results_dir = os.path.join(current_dir, dir_name)
        if os.path.exists(ray_results_dir):
            return ray_results_dir
        else:
            print(f"Lege Ordner an: {dir_name}")
            os.makedirs(ray_results_dir) 
            return ray_results_dir

    
    scheduler = tune.schedulers.ASHAScheduler(
        max_t = params.num_epochs, 
        grace_period = params.asha_grace_period, 
        reduction_factor = params.asha_reduction_factor
    )
    tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(ray_function),
                resources={"cpu": 1, "gpu": 0}), # CPUs / GPUs pro Trial
            tune_config=tune.TuneConfig(
                # metric="val_loss",
            metric="mean_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=params.ray_samples,
            # trial_name_creator=,
            trial_dirname_creator=ray_dir_name,
            ),
        run_config=ray.train.RunConfig(storage_path=create_storage_path('./Ray_results'),progress_reporter=reporter),
        param_space=params.hpo_config)
    
    ResultGrid = tuner.fit()
    best_result=ResultGrid.get_best_result(metric="mean_loss", mode="min")
    results_dic={'results': ResultGrid.get_dataframe(),
          'num_epochs': params.num_epochs,
          'best_trial_id': best_result.metrics["trial_id"]}
    results_file=open(Path(best_result.path,"..","HPO_results"),'wb')
    pickle.dump(results_dic,results_file)
    results_file.close()

class ray_hpo():
    def __init__(self, device,cwd):
        self.device = device
        self.cwd=cwd
        
    def train_fcn(self, config):
        pinn=pinn_cl(self.cwd)
        pinn.initParams()
        #current hpo config        
        pinn.n_neurons = config['n_neurons']
        pinn.n_hidden = config['n_hidden']
        pinn.lr_init = config['lr_init']
        if pinn.pinn_flag and pinn.ddpinn_flag:
            pinn.ddpinn_props["n_ansatz"] = config['n_ansatz']
        if pinn.pinn_flag==False:
            pinn.dropout_GRU=config['dropout_GRU']
        
        # data stuff
        pinn.loadTrainData()
        pinn.loadTestData()
        pinn.createDataloaderNoPinn()
        pinn.initTraining()
        pinn.train()