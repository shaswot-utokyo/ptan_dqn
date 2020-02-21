#!/bin/bash


for f in ./reed_job_temp_summary/*; do
    echo "REEDBUSH: Processing $f file..."
    # take action on each file. $f store current file name
    jobid=$(echo "${f##*.}" | cut -c 2-)
    echo $jobid
    printf "\n" >> $f
    rbstat -s -H $jobid >> $f 
done

