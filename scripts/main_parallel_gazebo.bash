#!/bin/bash

# Parameters: port
# 	      strategy

cd ..
cd devel && source setup.bash && cd ../
cd scripts

export ROS_MASTER_URI=http://localhost:1135$1
export GAZEBO_MASTER_URI=http://localhost:1134$1

roslaunch uav_gazebo simple_world.launch & PID_GAZEBO=$!
sleep 10
echo ${PID_GAZEBO}

strat=$1
isolation=$2
noises=(0 0.001 0.002 0.003 0.004 0.005)
# noises=(0.004 0.005)
time=`date +%s`
for noise in ${noises[@]}; do
	./multiple_run.bash $strat $isolation ${noise} 
done
kill -9 ${PID_GAZEBO}
