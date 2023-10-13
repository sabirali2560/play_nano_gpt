#!/bin/bash

num_scripts=(1 2 4 8)
for i in "${num_scripts[@]}";
do

    rm -rf output_fine_tune_$i
done    