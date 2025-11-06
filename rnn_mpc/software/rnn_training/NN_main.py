"""
--------------------------------------------------------------------------------
  Author:       M.Sc. Hendrik Schäfke
  Institution:  Institute of Mechatronic Systems, Leibniz Universität Hannover
  Date:         12.09.2024
  Description:  This script performs hyperparameter optimization for neural 
             networks using Ray Tune. It defines and optimizes the parameters 
             of two models: GRU and LSTM.
--------------------------------------------------------------------------------
"""
#%% Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
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
import NN_fcn 

#%% functions
def ray_function(config):
    ray_train = NN_fcn.ray_HyperParamOptimization(device, val_ratio, test_ratio, num_epochs, plot_interval, plot_seconds, warmup_batches, ray_flag)
    ray_train.train_fcn(config)

def ray_dir_name(trial):
    name="trial_"+str(trial.trial_id)
    return name

def create_storage_path(dir_name):
    current_dir = os.getcwd()
    ray_results_dir = os.path.join(current_dir, dir_name)
    if os.path.exists(ray_results_dir):
        return ray_results_dir
    else:
        print(f"Create folder: {dir_name}")
        os.makedirs(ray_results_dir)
        return ray_results_dir


#%% parameter
device = "cpu"
val_ratio = 0.7
test_ratio = 0.3
num_epochs = 300
plot_interval = 100
plot_seconds = 200
warmup_batches = 2
num_samples = 500
grace_period = 50
reduction_factor = 2
dir_name = "ray_results"
ray_flag = True

## Reproducibility of several runs
# torch.manual_seed(0)
# np.random.seed(0)

#%% config for HPO
# GRU
config = {
    "frequency_Hz" : 5,
    "model" : "GRU",
    "lr_init": tune.uniform(0.0001,0.001),
    "lr_factor": 0.6,
    "lr_patience" : 10,
    "step": 7,
    "sequence_length" : 200,
    "scaler_fn":  "MinMaxScaler",
    "neglegt_pdyn": True,
    "neglegt_qdyn": True,
    "n_aktoren": 5,
    "window": tune.choice([100]),
    "prediction" : tune.choice([20]),
    "batch_size": tune.choice([2,4,8,16,32]),
    "dropout" : tune.uniform(0.2,0.5),
    "num_layer" : tune.choice([1,2,3]),
    "hidden_dim" : tune.randint(16,64),
    "train_on_ramp" : False ,
}

# LSTM
config = {
    "frequency_Hz" : 5,
    "model" : "LSTM",
    "lr_init": tune.uniform(0.0001,0.001),
    "lr_factor": 0.6, 
    "lr_patience" : 10,
    "step": 7,
    "sequence_length" : 200,
    "scaler_fn": "MinMaxScaler",
    "neglegt_pdyn": True,
    "neglegt_qdyn" : True,
    "n_aktoren": 5,
    "window": tune.choice([100]),
    "prediction" : tune.choice([20]),
    "batch_size": tune.choice([2,4,8,16,32]),
    "dropout" : tune.uniform(0.2,0.5),
    "num_layer" : tune.choice([1,2,3]),
    "hidden_dim" : tune.randint(16,64),
    "train_on_ramp" : True ,
}

#%% start Raytune
if ray_flag == True:
    scheduler = ASHAScheduler(max_t = num_epochs, grace_period=grace_period, reduction_factor=reduction_factor)
    reporter = CLIReporter(max_progress_rows=100, max_report_frequency=1800)
    ray.shutdown()
    ray.init()

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(ray_function),
            resources={"cpu": 4, "gpu": 0}), # CPUs / GPUs per trial
        tune_config=tune.TuneConfig(
            metric="error_val_deg",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            trial_dirname_creator=ray_dir_name,
            ),
        run_config=train.RunConfig(storage_path=create_storage_path(dir_name)),
        param_space=config,)
    result_grid: ResultGrid = tuner.fit()

    # safe HPO parameter in table
    df = result_grid.get_dataframe() 
    output_dir = result_grid.experiment_path
    df.to_csv(os.path.join(output_dir, 'Hyperparameter.csv'),index=False, header=True, sep=';', decimal=',')