#!/bin/bash

for i in `seq 1 14`
do
    nohup python -u ./lucky_choice_santa.py > running_output_$i &
    sleep 1
done

