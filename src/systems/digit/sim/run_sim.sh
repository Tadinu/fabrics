#!/bin/bash
./bin/ar-control bin/lowlevelapi_stand.toml &
source ws/devel/setup.bash 
roslaunch digit_llapi main_llapi_sim.launch
