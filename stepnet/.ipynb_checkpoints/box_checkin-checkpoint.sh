#!/bin/sh  
while true  
do  
rclone copy /home/laura/code/multitask-nets/stepnet/data/crystals/ /home/laura/data/rnn/multitask/crystals/
rclone copy /home/laura/code/multitask-nets/highd_inputs/data/crystals/highd_inputs /home/laura/data/rnn/multitask/crystals/highd_inputs
rclone copy /home/laura/data/rnn/multitask/crystals stanford_box:multitask/crystals --transfers 10 --tpslimit 10 -P
sleep 300
done
