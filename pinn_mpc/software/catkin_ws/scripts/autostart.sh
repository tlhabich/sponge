#!/bin/bash
INIT1="/opt/ros/noetic/setup.bash"
INIT2="/home/ec/app_interface/ros_install/scripts/source_ros_install.sh"
INITCMD="source $INIT1 && source $INIT2"

tmux new-session -d -s app
tmux send-keys "top" C-m
tmux split-window -h 
tmux send-keys "watch -n1 \"dmesg | tail -n60\"" C-m
tmux split-window -v 
tmux send-keys "$INITCMD" C-m "roscore" C-m
tmux select-pane -t 0 
tmux split-window -v 
tmux send-keys "$INITCMD" C-m "/home/ec/rtmdl/test_bench" C-m
