#!/bin/bash

##SBATCH --job-name=test
##SBATCH --output=out_test.dat

# Output/Error log directory
#SBATCH -o ./ist_job_temp/ist_job.out.%j
#SBATCH -e ./ist_job_temp/ist_job.err.%j

# use partition 'p'
#SBATCH -p p
##SBATCH -p v

# use gpu resource
#SBATCH  --gres=gpu:1

# allocate run time
#SBATCH --time=10:00:00

# Number of nodes
#SBATCH -N 1

# Number of tasks
#SBATCH -n 1

# Number of cores per task
#SBATCH -c 12

# Memory per node
#SBATCH --mem 12GB


# get file with the list of seeds
seed=$1
experiment=$2
# python dqn_test.py --cuda --seed=$1 $2
# python dqn_basic.py --cuda --seed=$1 $2
# python dqn_srg.py --cuda --seed=$1 $2
# python dqn_nstep.py --cuda --seed=$1 --nsteps=3 $2
# python dqn_nstep_double_dueling.py --cuda --seed=$1 --nsteps=3 --double --dueling $2
python dqn_nstep_double_dueling_srg.py --cuda --seed=$1 --nsteps=3 --double --dueling $2



sstat -p -j $SLURM_JOB_ID.batch --format=JobID,MaxRss,MaxVMSize,NTasks,ConsumedEnergy
##sacct -j %j --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

## USAGE
## sbatch ist_single_job.sh <seed> <envname>
## sbatch ist_single_job.sh 1234 mypong

