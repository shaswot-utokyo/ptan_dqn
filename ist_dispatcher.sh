#!/bin/bash

# get file with the list of seeds
filename=$1
experiment=$2
echo "Seedfile = $filename"
while IFS= read -r line
do
    ## reading each line
    echo "$line"
    sbatch ist_single_job.sh "$line" "$2"
    sleep 5 
done < "$filename"

## USAGE
## Sends out jobs to slurm. Each job contains one seed run of one experiment
# ./ist_dispatcher.sh seed_list.dat pong
