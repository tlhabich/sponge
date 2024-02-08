# Test Bench of SPONGE

The test bench communication is realized using EtherCAT protocol and the corresponding open-source tool [EtherLab](https://etherlab.org/) with an added [external-mode patch](https://github.com/SchapplM/etherlab-examples). Note that this enables (hard) real-time system control, data acquisition, online visualization and alteration of settings during runtime via Simulink. A ROS integration is also possible.

## Requirements
- Matlab R2018b (tested with it, update to newer versions pending)
- Real-time computer (RT-PC) with EtherCAT ([SETUP_RTPC.MD](https://github.com/SchapplM/etherlab-examples/blob/master/SETUP_RTPC.MD) and [SETUP_ETHERCAT.MD](https://github.com/SchapplM/etherlab-examples/blob/master/SETUP_ETHERCAT.MD)).
- Development computer (DEV-PC) with EtherLab (also [SETUP_ETHERCAT.MD](https://github.com/SchapplM/etherlab-examples/blob/master/SETUP_ETHERCAT.MD))
- **TO DO:** Describe I2C requirements and upload necessary files

## Usage
1. DEV-PC: Initialize parameters and open Simulink model via ``init.m``
2. DEV-PC: If necessary, modify Simulink model
3. DEV-PC: Compile/Build model via ``Ctrl+B``
4. DEV-PC: Compile ROS-Workspace and copy to RT-PC via ``$ ./build.sh && ./sync.sh``
5. RT-PC: Connect to RT-PC via SSH and run the following commands: ``ec_start`` (start EtherCAT master) and ``$ ~/app_interface/ros_install/scripts/autostart.sh && tmux attach-session -t app`` (start compiled model)
6. DEV-PC: Start external model in Simulink model to visualize/record data or alter settings
7. After the experiment on RT-PC: ``Ctrl+C`` in tmux windows, ``$ tmux kill-session`` and ``$ ec_stop`` to stop the EtherCAT master
8. DEV-PC: Postprocessing of the recorded data via ``postprocess.m``
