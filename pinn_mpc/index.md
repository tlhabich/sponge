---
title: MPC with PINNs
nav_order: 6
---

# Physics-Informed Neural Networks for Learning and Control
<p align="center">
<img src="images/../../images/pinn_cover.png" width=600>
</p>

This page describes the implementation of a learning-based nonlinear model predictive control (NMPC) using physics-informed neural networks (PINNs), which is experimentally validated with SPONGE. PINNs are trained in PyTorch, and their hyperparameters are optimized via ASHA. One PINN is integrated into a ROS package (``sponge_mpc``) and used as model within NMPC. For this purpose, CasADi is used in a ROS service (C++ node) which communicates with Simulink, enabling its use on the real-time hardware. The test-bench software is based on the [originally test-bench software](https://tlhabich.github.io/sponge/test_bench/). It is therefore recommended that you familiarize yourself with this.

**The code for PINN training, hyperparameter optimization and learning-based NMPC with PINNs can be found in the [git repository](https://github.com/tlhabich/sponge/tree/main/pinn_mpc/software). 13 open-source real-world datasets of SPONGE with five actuators can be found [here](https://repo.uni-hannover.de/items/0108821f-2ab3-4d28-a5af-420ff2548277).**

## Additional Requirements
- [CasADi](https://github.com/casadi/casadi/wiki/InstallationLinux) installed as a source build with IPOPT solver
- [json.hpp](https://github.com/nlohmann/json) copied to [include folder](https://github.com/tlhabich/sponge/tree/main/pinn_mpc/software/catkin_ws/src/sponge_mpc/include)

## Usage
1. Set up the test bench following these [instructions](https://tlhabich.github.io/sponge/test_bench/). The ROS-interface is used, which is explained [here](https://github.com/SchapplM/etherlab-examples)
2. Dev-PC: Initialize parameters and open Simulink model via ``init.m``
3. Dev-PC: If necessary, modify Simulink model
4. Dev-PC: Compile the model by pressing ``Ctrl+b``
5. Dev-PC: Compile ROS-Workspace and copy to RT-PC via ``$ ./build.sh && ./sync.sh``
6. Connect to RT-PC via SSH and run the following commands on RT-PC: ``$ sudo /etc/init.d/ethercat start`` (start EtherCAT master) and ``$ ~/app_interface/ros_install/scripts/autostart.sh && tmux attach-session -t app`` (start compiled model)
7. Dev-PC: Launch ROS service for PINN-based MPC via ``roslaunch sponge_mpc sponge_mpc.launch``
8. Dev-PC: Start external mode in Simulink model via ``Connect To Target`` to visualize/record data or alter settings (such as starting the MPC experiment)
9. After the experiment on RT-PC: ``Ctrl+c`` in tmux window, ``$ tmux kill-session -t app`` and ``$ sudo /etc/init.d/ethercat stop`` to stop the EtherCAT master

## Citing
The paper is [freely available](https://arxiv.org/abs/2502.01916) via arXiv. If you use parts of this project for your research, please cite the following publication:
```
Generalizable and Fast Surrogates: Model Predictive Control of Articulated Soft Robots using Physics-Informed Neural Networks
T.-L. Habich, A. Mohammad, S. F. G. Ehlers, M. Bensch, T. Seel, and M. Schappler
IEEE Transactions on Robotics (T-RO) 2025
DOI: 10.1109/TRO.2025.3631818
```
