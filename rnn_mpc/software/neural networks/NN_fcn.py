"""
--------------------------------------------------------------------------------
  Author:       M.Sc. Hendrik Schäfke
  Institution:  Institute of Mechatronic Systems, Leibniz Universität Hannover
  Date:         12.09.2024
  Description:  This script defines the functions and classes required for 
             the NN_main script. It includes data loading, preprocessing, 
             defining various neural network architectures, training routines, 
             and saving the results.
--------------------------------------------------------------------------------
"""
#----------------------------
#%% Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.io
import time
import torch # PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import tempfile
from pathlib import Path
from torchinfo import summary
import copy
from timeit import default_timer as timer
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import ExperimentAnalysis
from ray.tune import ResultGrid
from ray.train import Result
from datetime import datetime

#----------------------------
#%% Read data
def loadData(pfad_data,n_aktoren=5, neglegt_pdyn=True, neglegt_qdyn=True, cut_sample=0,):
    data_raw = pd.read_csv(pfad_data, sep = ',', decimal = '.')
    data_raw = data_raw.astype(np.float32)
    data_raw = data_raw.to_numpy()
    if cut_sample>0 and cut_sample<data_raw.shape[0]:
        data_raw=data_raw[:cut_sample,:]
    data_raw=np.delete(data_raw,[0,1,2],1) # data_raw structure: q1,q2,..,qd1,qd2,..,p11,p12,p21,p22,..,,p11d,p12d,p21d,p22d,..

    if n_aktoren==3: # Delete data from 4.5 actuators if only 3 have been moved
        data_raw=np.delete(data_raw,[3,4,8,9,16,17,18,19,26,27,28,29],1) 
    xdim=4*n_aktoren
    udim = 2*n_aktoren

    if neglegt_pdyn == True: # Pressure dynamics are ignored -> no p as a state
        data_raw=np.delete(data_raw,[np.arange(n_aktoren*2, n_aktoren*4)],1)
        xdim = xdim - 2*n_aktoren

    if neglegt_qdyn == True: # Angular velocity is ignored as state
        data_raw=np.delete(data_raw,[np.arange(n_aktoren, n_aktoren*2)],1)
        xdim = xdim - n_aktoren

    X_raw=data_raw
    Y_raw=data_raw[:,:xdim]
    return X_raw,Y_raw, xdim, udim

def split_scale_Data(X_raw, Y_raw, val_ratio, x_scaler, y_scaler):
    # scale
    Xs = x_scaler.transform(X_raw)
    Ys = y_scaler.transform(Y_raw)

    Xs = Xs[:-1]
    Ys = Ys[1:]

    # Train Val Split
    nstep = Xs.shape[0] # number of samples
    cut_index = np.int32(nstep*val_ratio)
    Xs_train = Xs[0:cut_index]
    Ys_train = Ys[0:cut_index]
    Xs_val = Xs[cut_index:]
    Ys_val = Ys[cut_index:]
    return Xs_train, Ys_train, Xs_val, Ys_val, 
    
def daten_einlesen(X_raw, Y_raw, xdim, udim, window, prediction, x_scaler, y_scaler, val_ratio, batch_size, shuffle, step=1):
    # scale
    Xs = x_scaler.transform(X_raw)
    Ys = y_scaler.transform(Y_raw)

    # Train Val Split
    nstep = Xs.shape[0]
    cut_index = np.int32(nstep*val_ratio)
    Xs_train = Xs[0:cut_index]
    Ys_train = Ys[0:cut_index]
    Xs_val = Xs[cut_index:]
    Ys_val = Ys[cut_index:]

    X_train = []
    Y_train = []
    for i in range(window,len(Xs_train)-prediction):
        X_train.append(Xs_train[i-window:i+prediction,:])
        Y_train.append(Ys_train[i:i+prediction])

    X_val = []
    Y_val = []
    for i in range(window,len(Xs_val)-prediction):
        X_val.append(Xs_val[i-window:i+prediction,:])
        Y_val.append(Ys_val[i:i+prediction])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)

    for i in range(0, len(X_train)):
        X_train[i][window:,:xdim] = X_train[i][window-1,:xdim]

    for i in range(0, len(X_val)):
        X_val[i][window:,:xdim] = X_val[i][window-1,:xdim]

    idx_down_train = [i for i in range(0,len(X_train),step)]
    idx_down_val = [i for i in range(0,len(X_val),step)]
    dataset_train = DatasetTorch(X_train[idx_down_train], Y_train[idx_down_train])
    dataset_val = DatasetTorch(X_val[idx_down_val], Y_val[idx_down_val])
    train_DL = DataLoader(dataset_train, batch_size, shuffle, drop_last=False)
    val_DL = DataLoader(dataset_val, batch_size , shuffle, drop_last=False)
    return train_DL, val_DL

class DatasetTorchSplit(Dataset):
    def __init__(self, Xs_data, Ys_data, sequence_length):
        self.Xs_data = torch.from_numpy(Xs_data).float()
        self.Ys_data = torch.from_numpy(Ys_data).float()
        self.Xs_data = self.Xs_data.split(sequence_length)
        self.Ys_data = self.Ys_data.split(sequence_length)
        self.Xs_data = self.Xs_data[:-1]
        self.Ys_data = self.Ys_data[:-1]
    def __len__(self):
        return len(self.Xs_data)
    def __getitem__(self,idx):
        Xs_data = self.Xs_data[idx]
        Ys_data = self.Ys_data[idx]
        return Xs_data, Ys_data

class DatasetTorch(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.Y_data = torch.from_numpy(Y_data).float()
    def __len__(self):
        return len(self.X_data)
    def __getitem__(self,idx):
        X_data = self.X_data[idx]
        Y_data = self.Y_data[idx]
        return X_data, Y_data
    
#----------------------------
#%% define neural networks
class GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, dropout):
        super(GRU, self).__init__()
        self.ident = "GRU"
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layer, batch_first = True, dropout = dropout )
        self.dense = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, ht):
        out, ht = self.gru(x, ht)
        out = out[:, -1, :]  # Select the output of the last time step
        out = self.dense(out)
        return out, ht

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, dropout):
        super(LSTM, self).__init__()
        self.ident = "LSTM"
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layer, batch_first = True, dropout = dropout )
        self.dense = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, ht, ct):
        out, (ht, ct) = self.lstm(x, (ht, ct))
        out = out[:, -1, :]  # Select the output of the last time step
        out = self.dense(out)
        return out, ht, ct
    

def save_NN(model, config, path, x_scaler, y_scaler, xdim=0, udim=0, total_epochs=0):
    # Save the parameters of the GRU mesh as a .mat file 
    # The weights are saved separately to simplify the transfer to Matlab/Simulink
    if model.ident == "GRU":
        weights = model.state_dict()
        num_layer = model.num_layer
        GRU_params = {
                        "weights": weights,
                        "hidden_dim": model.hidden_dim,
                        "num_layer": model.num_layer,
                        "n_aktoren": config["n_aktoren"],
                        "lr_init": config["lr_init"],
                        "sequence_length": config["sequence_length"],
                        "neglegt_pdyn": config["neglegt_pdyn"],
                        "neglegt_qdyn": config["neglegt_qdyn"],
                        "config": config,
                        "xdim": xdim,
                        "udim": udim,
                        "total_epochs": total_epochs,
                        "x_scaler_min": x_scaler.data_min_,
                        "x_scaler_max": x_scaler.data_max_,
                        "x_scaler_feature_range": x_scaler.feature_range,
                        "y_scaler_min": y_scaler.data_min_,
                        "y_scaler_max": y_scaler.data_max_,
                        "y_scaler_feature_range": x_scaler.feature_range,
                        }
        scipy.io.savemat(os.path.join(path,f"GRU_params.mat"),GRU_params, long_field_names=True) #, oned_as='column')
        GRU_weights = {}
        for layer in range(num_layer):
            w_ih = getattr(model.gru, f'weight_ih_l{layer}').chunk(3, 0)
            b_ih = getattr(model.gru, f'bias_ih_l{layer}').chunk(3, 0)
            w_hh = getattr(model.gru, f'weight_hh_l{layer}').chunk(3, 0)
            b_hh = getattr(model.gru, f'bias_hh_l{layer}').chunk(3, 0)
            weights_names = [f'w_ir_l{layer}', f'w_iz_l{layer}', f'w_in_l{layer}',
                            f'b_ir_l{layer}', f'b_iz_l{layer}', f'b_in_l{layer}',
                            f'w_hr_l{layer}', f'w_hz_l{layer}', f'w_hn_l{layer}',
                            f'b_hr_l{layer}', f'b_hz_l{layer}', f'b_hn_l{layer}']
            for name, matrix in zip(weights_names, w_ih + b_ih + w_hh + b_hh):
                GRU_weights[name] = matrix.data.detach().numpy()
        weight_linout = model.dense.weight
        GRU_weights['weight_linout']=weight_linout.data.detach().numpy()
        bias_linout = model.dense.bias
        GRU_weights['bias_linout']=bias_linout.data.detach().numpy()
        scipy.io.savemat(os.path.join(path,f"GRU_weights.mat"),GRU_weights, long_field_names=True) #, oned_as='column')

def save_config_txt(config, path):
    with open(os.path.join(path,f"config.txt"), 'w') as f:  
                    for key, value in config.items():  
                        f.write('%s:%s\n' % (key, value))

#%% Training
def NNstep(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training:bool,
        window,
        xdim,
        udim,
        device = "cpu",
        output_pred = False):

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_sum = 0
    for idx, (X_data, Y_data) in enumerate(dataloader):
        X_data, Y_data = X_data.to(device).float(), Y_data.to(device).float()
        X_pred = X_data.detach().clone()
        Y_pred = Y_data.detach().clone()
        batch_size = X_data.shape[0]
        input_dim = X_data.shape[2]
        sequence_length = X_data.shape[1]
        
        if model.ident == "GRU":
            if idx == 0:
                ht = torch.zeros(model.num_layer , batch_size, model.hidden_dim)
                x_k = X_data[:,0,:xdim].reshape((1, 1, xdim))
            else:
                ht = ht.detach()
                x_k = x_k.detach()

            for i in range(sequence_length):
                x_k = x_k.reshape(1,1,xdim)
                u_k = X_pred[:,i,xdim:].reshape(1,1,udim)
                Xin = torch.cat((x_k, u_k), dim=2)
                x_k , ht = model(Xin, ht)
                Y_pred[:,i,:] = x_k

        if model.ident == "LSTM":
            if idx == 0:
                ht = torch.zeros(model.num_layer , batch_size, model.hidden_dim)
                ct = torch.zeros(model.num_layer , batch_size, model.hidden_dim)
                x_k = X_data[:,0,:xdim].reshape((1, 1, xdim))
            else:
                ht = ht.detach()
                ct = ct.detach()
                x_k = x_k.detach()

            for i in range(sequence_length):
                x_k = x_k.reshape(1,1,xdim)
                u_k = X_pred[:,i,xdim:].reshape(1,1,udim)
                Xin = torch.cat((x_k, u_k), dim=2)
                x_k , ht, ct = model(Xin, ht, ct)
                Y_pred[:,i,:] = x_k

        loss = loss_fn(Y_pred, Y_data)
        loss_sum += loss.item()
            
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if output_pred == True:
            if idx==0:
                X_data_out = X_data.detach().squeeze()
                Y_data_out = Y_data.detach().squeeze()
                Y_pred_out = Y_pred.detach().squeeze()
            else:
                X_data_out = torch.vstack((X_data_out,X_data.detach().squeeze()))
                Y_data_out = torch.vstack((Y_data_out,Y_data.detach().squeeze()))
                Y_pred_out = torch.vstack((Y_pred_out,Y_pred.detach().squeeze()))
        else:
            X_data_out=[]
            Y_data_out=[]
            Y_pred_out=[]
                    
    loss_sum = loss_sum / len(dataloader)
    return loss_sum , X_data_out, Y_data_out, Y_pred_out

def NNstep_window(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training:bool,
        window,
        prediction,
        xdim,
        udim,
        device = "cpu",):

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_sum = 0
    for idx, (X_data, Y_data) in enumerate(dataloader):
        X_data, Y_data = X_data.to(device).float(), Y_data.to(device).float()
        X_pred = X_data.detach().clone()
        Y_pred = Y_data.detach().clone()
        batch_size = X_data.shape[0]
        input_dim = X_data.shape[2] # = xdim + udim

        if model.ident == "GRU":
            ht = torch.zeros(model.num_layer , batch_size, model.hidden_dim)
            for i in range(window,window+prediction):
                if i == window:
                    Xin = X_pred[:,i-window:i].reshape(batch_size, window, input_dim)
                    x_k, ht = model(Xin, ht)
                    Y_pred[:,i-window,:] = x_k
                else:
                    u_k = X_pred[:,i,xdim:].reshape(batch_size, 1, udim)
                    x_k = x_k.unsqueeze(1)
                    Xin = torch.cat((x_k, u_k), dim=2)
                    x_k , ht = model(Xin, ht)
                    Y_pred[:,i-window,:] = x_k

        if model.ident == "LSTM":
            ht = torch.zeros(model.num_layer , batch_size, model.hidden_dim)
            ct = torch.zeros(model.num_layer , batch_size, model.hidden_dim)
            for i in range(window,window+prediction):
                if i == window:
                    Xin = X_pred[:,i-window:i].reshape(batch_size, window, input_dim)
                    x_k, ht, ct = model(Xin, ht, ct)
                    Y_pred[:,i-window,:] = x_k
                else:
                    u_k = X_pred[:,i,xdim:].reshape(batch_size, 1, udim)
                    x_k = x_k.unsqueeze(1)
                    Xin = torch.cat((x_k, u_k), dim=2)
                    x_k , ht, ct = model(Xin, ht, ct)
                    Y_pred[:,i-window,:] = x_k

        loss = loss_fn(Y_pred, Y_data)
        loss_sum += loss.item()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
    loss_sum = loss_sum / len(dataloader)
    return loss_sum

#---------
def train_epochs(model: torch.nn.Module,
        train_DL: torch.utils.data.DataLoader,
        val_DL: torch.utils.data.DataLoader,
        test_DL: torch.utils.data.DataLoader,
        val_DL_sq: torch.utils.data.DataLoader,
        test_DL_sq: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        scheduler,
        xdim,
        udim,
        plot_interval,
        config,
        n_aktoren,
        x_scaler,
        y_scaler,
        ray_flag = False,
        neglegt_pdyn = True,
        neglegt_qdyn = True,
        plot_seconds = 200,
        device = "cpu"):

    # Init
    results = {"number_epoch": [],"train_loss": [],"val_loss": [], "test_loss": [], "time_per_epoch": []}
    current_date = datetime.now().strftime('Trial_%Y-%m-%d_%H-%M')
    results_folder = os.path.join('standalone_results', current_date)

    for epoch in range(epochs):
        epoch_start_timer = timer()

        train_loss = NNstep_window(model, train_DL, loss_fn, optimizer, training=True, window=config["window"], prediction=config["prediction"], xdim=xdim, udim=udim, device=device, train_delta = train_delta) # train step
        val_loss = NNstep_window(model, val_DL, loss_fn, optimizer, training=False, window=config["window"], prediction=config["prediction"], xdim=xdim, udim=udim, device=device,  train_delta = train_delta) # val step
        test_loss = NNstep_window(model, test_DL, loss_fn, optimizer, training=False, window=config["window"], prediction=config["prediction"], xdim=xdim, udim=udim, device=device,  train_delta = train_delta) # test step
 
        val_loss_sq, Xs_data_out_val, Ys_data_out_val, Ys_pred_out_val = NNstep(model, val_DL_sq, loss_fn, optimizer, training=False, window=config["window"], xdim=xdim, udim=udim, device=device, output_pred=True, train_delta = train_delta) # val step
        test_loss_sq, Xs_data_out_test, Ys_data_out_test, Ys_pred_out_test = NNstep(model, test_DL_sq, loss_fn, optimizer, training=False, window=config["window"], xdim=xdim, udim=udim, device=device, output_pred=True, train_delta=train_delta) # test step

        epoch_end_timer = timer()
        time_per_epoch = epoch_end_timer - epoch_start_timer
        results["number_epoch"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["test_loss"].append(test_loss)
        results["time_per_epoch"].append(time_per_epoch)
        print(
            f"Epoche: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"time_per_epoch: {time_per_epoch:.4f} s")
        
        scheduler.step(val_loss)
        parameters = summary(model, verbose=0).trainable_params

        # scale inverse
        X_data_out_val = x_scaler.inverse_transform(Xs_data_out_val)
        Y_data_out_val = y_scaler.inverse_transform(Ys_data_out_val)
        Y_pred_out_val = y_scaler.inverse_transform(Ys_pred_out_val)
        error_val_deg = np.mean(np.abs(np.degrees(Y_data_out_val - Y_pred_out_val)))

        X_data_out_test = x_scaler.inverse_transform(Xs_data_out_test)
        Y_data_out_test = y_scaler.inverse_transform(Ys_data_out_test)
        Y_pred_out_test = y_scaler.inverse_transform(Ys_pred_out_test)
        error_test_deg = np.mean(np.abs(np.degrees(Y_data_out_test - Y_pred_out_test)))
        
        if ray_flag == True:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                if (epoch+1) % plot_interval == 0 and epoch > 0 or epoch == epochs-1:
                    trial_name = ray.train.get_context().get_trial_name()
                    torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'train_loss':train_loss,'val_loss':val_loss,'test_loss':test_loss,'config' : config, 'trial_name':trial_name}, os.path.join(temp_checkpoint_dir, 'checkpoint.pt'))
                    save_NN(model, config, path=temp_checkpoint_dir, x_scaler=x_scaler, y_scaler=y_scaler, xdim=xdim, udim=udim, total_epochs=epoch)
                    save_config_txt(config, path=temp_checkpoint_dir)
                    plot_loss_curves(results,"Loss", path=temp_checkpoint_dir)                
                    plot_pred_all(X_data_out_val, Y_data_out_val, Y_pred_out_val, config, n_aktoren, plot_seconds=plot_seconds, loss=val_loss, path=temp_checkpoint_dir, name='val')
                    plot_pred_all(X_data_out_test, Y_data_out_test, Y_pred_out_test, config, n_aktoren, plot_seconds=plot_seconds, loss=test_loss, path=temp_checkpoint_dir, name='test')
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report({"val_loss":val_loss, "train_loss":train_loss, "test_loss":test_loss, "error_val_deg": error_val_deg, "error_test_deg":error_test_deg, "parameters":parameters}, checkpoint = checkpoint) # veraltet: tune.report(val_loss = val_loss, iterations = epoch) 
                else:
                    train.report({"val_loss":val_loss, "train_loss":train_loss, "test_loss":test_loss, "error_val_deg": error_val_deg, "error_test_deg":error_test_deg, "parameters":parameters})

#----------------------------
#%% Plots
def plot_loss_curves(results, title, path):
    x = results["number_epoch"]
    time_per_epoch_mean = np.mean(results["time_per_epoch"])
    fig = plt.figure(figsize=(15, 7))
    plt.plot(x, results["train_loss"], label='train_loss')
    plt.plot(x, results["val_loss"], label='val_loss')
    plt.plot(x, results["test_loss"], label='test_loss')
    plt.title(f"Zeit pro Epoche: {time_per_epoch_mean:.2f} s")
    plt.xlabel('Epochen')
    plt.gca().set_xlim(left=0)
    plt.ylabel('MSE Loss')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(path,f"Loss.pdf"))
    plt.close('all')

def plot_pred_all(X_data, Y_data, Y_pred, config, n_aktoren, plot_seconds, loss, path, name=''):
    sample_rate = config["frequency_Hz"]
    sampling_interval = 1/sample_rate 
    samples = int(plot_seconds * (1/sampling_interval))
    t = np.arange(0, len(X_data)) * sampling_interval
    error_deg = np.mean(np.abs(np.degrees(Y_data - Y_pred)))
    imesblau   = "#00509B" # [0 80 155 ]/255; # imsblau
    imesorange = "#E77B29" # [231 123 41 ]/255; # imesorange
    imesgruen  = "#C8D317" # [200 211 23 ]/255; # imesgrün

    fig, axs = plt.subplots(n_aktoren, 1, figsize=(7.1 , 5), sharex=True)
    for aktor in range(n_aktoren):
        axs[aktor].plot(t[:samples], np.degrees(Y_data[:samples, aktor]), label="measured", color=imesblau)
        axs[aktor].plot(t[:samples], np.degrees(Y_pred[:samples, aktor]), label="predicted", color=imesorange)
        axs[aktor].set_xlim(left=0, right=t[:samples][-1])
        axs[aktor].set_ylim(bottom=-25, top=25)
        axs[aktor].grid()
        axs[aktor].set_ylabel(f"q{aktor + 1} in Grad")
        if aktor==0:
            axs[aktor].legend(loc="upper right")
    axs[-1].set_xlabel("Zeit in s")
    fig.text(0.5, 0.97, f'Durchschnittlicher Fehler: {round(error_deg, 3)}°', ha='center')
    fig.tight_layout()
    fig.savefig(os.path.join(path,f"Pred_all_{name}.pdf"))
    plt.close('all')

class ray_HyperParamOptimization():
    def __init__(self, device, val_ratio, test_ratio, num_epochs, plot_interval, plot_seconds, ray_flag, server_flag):
        self.device = device
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_epochs = num_epochs
        self.plot_interval = plot_interval
        self.plot_seconds = plot_seconds
        self.ray_flag = ray_flag
        self.server_flag = server_flag
    def train_fcn(self, config):

        if config['scaler_fn'] == "MinMaxScaler":
            scaler_fn = MinMaxScaler(feature_range=(-1,1))
        elif config['scaler_fn'] == "StandardScaler":
            scaler_fn = StandardScaler()

        x_scaler = copy.deepcopy(scaler_fn)
        y_scaler = copy.deepcopy(scaler_fn)
        
        pfad_aktuell = os.getcwd()
        if config['n_aktoren'] ==3:
            datensatz_sprung = "DS15_3Aktoren_11122023_FF30min_0grad_0gramm_Spruenge_fil_downsampled_25Hz.csv"
            datensatz_ramp = "DS15_3Aktoren_11122023_FF30min_0grad_0gramm_Rampen_fil_downsampled_25Hz.csv"
        
        elif config['n_aktoren']==5:
            frequency = config.get('frequency_Hz')
            try:
                datensatz_sprung = f"DS15_5Actuators_FF30min_0grad_0grams_Steps_20240613_fil_downsampled_{frequency}Hz.csv"
                datensatz_ramp = f"DS15_5Actuators_FF30min_0grad_0grams_Ramps_20240613_fil_downsampled_{frequency}Hz.csv"

            except TypeError:
                print("dataset not found")

            else:
                print("dataset not found")


        elif self.server_flag == 0: ## Laptop
            if config["train_on_ramp"] == False:
                pfad_train = Path(pfad_aktuell,'messdaten',datensatz_sprung)
                pfad_test = Path(pfad_aktuell,'messdaten',datensatz_ramp)
            elif config["train_on_ramp"] == True:
                pfad_train = Path(pfad_aktuell,'messdaten',datensatz_ramp)
                pfad_test = Path(pfad_aktuell,'messdaten',datensatz_sprung)
        X_raw, Y_raw, xdim, udim = loadData(pfad_train, config['n_aktoren'], config["neglegt_pdyn"], config["neglegt_qdyn"])
        X_raw_test, Y_raw_test, xdim, udim = loadData(pfad_test, config['n_aktoren'], config["neglegt_pdyn"], config["neglegt_qdyn"])

        x_scaler.fit(X_raw)
        y_scaler.fit(Y_raw)

        train_DL, val_DL = daten_einlesen(X_raw, Y_raw, xdim, udim, config['window'], config['prediction'], x_scaler, y_scaler, self.val_ratio, config['batch_size'], True, config["step"])
        test_DL, _ = daten_einlesen(X_raw_test, Y_raw_test, xdim, udim, config['window'], config['prediction'], x_scaler, y_scaler, self.test_ratio, config['batch_size'], False, config["step"])

        Xs_train, Ys_train, Xs_val, Ys_val = split_scale_Data(X_raw, Y_raw, self.val_ratio, x_scaler, y_scaler)
        Xs_test, Ys_test, Xs_test_, Ys_test_ = split_scale_Data(X_raw_test, Y_raw_test, self.test_ratio, x_scaler, y_scaler)
        
        val_DS_sq = DatasetTorchSplit(Xs_val, Ys_val, config['sequence_length'])
        test_DS_sq = DatasetTorchSplit(Xs_test, Ys_test, config['sequence_length'])

        val_DL_sq = DataLoader(val_DS_sq, batch_size=1, shuffle=False)
        test_DL_sq = DataLoader(test_DS_sq, batch_size=1, shuffle=False)
        
        input_dim = xdim+udim
        output_dim = xdim

        # initialize models
        if config["model"] == "GRU":
            model = GRU(input_dim, output_dim, config['hidden_dim'], config['num_layer'], config['dropout'])
        elif config["model"] == "LSTM":
            model = LSTM(input_dim, output_dim, config['hidden_dim'], config['num_layer'], config['dropout'])
        model = model.to(self.device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = config['lr_init'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = config['lr_factor'] , patience = config['lr_patience'], verbose = True)
        train_epochs(model, train_DL, val_DL, test_DL, val_DL_sq, test_DL_sq, loss_fn, optimizer, self.num_epochs, lr_scheduler, xdim, udim, self.plot_interval, config, config['n_aktoren'], x_scaler, y_scaler, self.ray_flag, config['train_delta'], config["neglegt_pdyn"], config["neglegt_qdyn"], self.plot_seconds, self.device)