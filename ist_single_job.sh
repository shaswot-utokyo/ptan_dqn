#!/bin/bash

##SBATCH --job-name=test
##SBATCH --output=out_test.dat

#SBATCH -o ./ist_job.out.%j
#SBATCH -e ./ist_job.err.%j

# use partition 'p'
#SBATCH -p p

# use gpu resource
#SBATCH  --gres=gpu:1

# allocate run time
#SBATCH --time=20:00:00

# Number of nodes
#SBATCH -N 1

# Number of tasks
#SBATCH -n 1

# Number of cores per task
#SBATCH -c 6

# Memory per node
#SBATCH --mem 12GB



# get file with the list of seeds
seed=$1
experiment=$2
# python dqn_basic.py --cuda --seed=$1 $2
python dqn_srg.py --cuda --seed=$1 $2


## USAGE
## sbatch ist_single_job.sh <seed> <envname>
## sbatch ist_single_job.sh 1234 mypong

