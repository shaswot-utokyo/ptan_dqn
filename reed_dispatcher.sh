#!/bin/bash

# get file with the list of seeds
filename=$1
experiment=$2
echo "Seedfile = $filename"
while IFS= read -r line
do
    ## reading each line
    echo "$line"
    qsub -v seed="$line",experiment="$2" reed_single_job.sh
    sleep 5 
done < "$filename"

## USAGE
## Sends out jobs to PBS. Each job contains one seed run of one experiment
