# -*- coding: utf-8 -*-
"""
DD-PINN/PINC or black-box model (FNN and RNN): training (standalone or hyperparameter optimization)
@author: TLH
"""
import os
import PINN_helper
import pathlib
import torch
import numpy as np
import random

cwd = pathlib.Path(os.getcwd())

# reproducibility of several runs
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# higher precision with float64
torch.set_default_dtype(torch.float64)

# init
pinn=PINN_helper.pinn_cl(cwd)
pinn.initParams()

if pinn.retrain_network[0]:
    PINN_helper.retrain(pinn.retrain_network[1],cwd,pinn.retrain_network[2],pinn.retrain_network[3])
else:   
    # data stuff
    pinn.loadTrainData()
    pinn.loadTestData()
    pinn.createDataloaderNoPinn()
        
    # training
    if pinn.hpo_flag:# HPO
        PINN_helper.start_hpo(pinn,cwd)      
    else:# standalone training
        pinn.createFolder(cwd)
        pinn.initTraining()
        pinn.train()
