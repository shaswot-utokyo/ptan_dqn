#!/bin/bash


for f in ./ist_job_temp_summary/*; do
    echo "Processing $f file..."
    # take action on each file. $f store current file name
    echo "${f##*.}" 
    printf "\n" >> $f
    sacct -j "${f##*.}" --format=JobID,state,start,elapsed,MaxRss,MaxVMSize,ncpus,nodelist >> $f
    #cat $f
done

