#!/bin/bash

for i in `seq 1 15`
do
    #nohup python -u ./lucky_choice_santa.py > running_output_$i &
    nohup ./optimize_main > running_output_cpp_$i &
    sleep 5
done

# sudo kill -9 $(pidof 进程名关键字)
# sudo kill -9 $(pidof optimize_main)

