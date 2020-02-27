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
#SBATCH --mem 32GB


# get file with the list of seeds
seed=$1
experiment=$2

# OLD JOBS
#################################################################################################
# python dqn_test.py --cuda --seed=$1 $2
# python dqn_basic.py --cuda --seed=$1 $2
# python dqn_srg.py --cuda --seed=$1 $2
# python dqn_nstep.py --cuda --seed=$1 --nsteps=3 $2
# python dqn_nstep_double_dueling.py --cuda --seed=$1 --nsteps=3 --double --dueling $2
# python dqn_nstep_double_dueling_srg.py --cuda --seed=$1 --nsteps=3 --double --dueling $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --dueling --srg=0.0001 $2
#################################################################################################

# NEW JOBS
# USAGE:
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=<seed_value> --nsteps=<rollout_length> --double --dueling --srg=<srg_ratio> <experiment>

# # SRG = OFF
# REED
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 $2

# IST
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double $2

# REED
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --dueling $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --dueling $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --dueling $2

# IST
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --dueling $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --dueling $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --dueling $2

# IST
# # SRG = 0.0001
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --srg=0.0001 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --srg=0.0001 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --dueling --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --dueling --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --dueling --srg=0.0001 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --dueling --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --dueling --srg=0.0001 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --dueling --srg=0.0001 $2

# REED
# # SRG = 0.0002
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --srg=0.0002 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --srg=0.0002 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --dueling --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --dueling --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --dueling --srg=0.0002 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --dueling --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --dueling --srg=0.0002 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --dueling --srg=0.0002 $2

# IST
# # SRG = 0.0005
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --srg=0.0005 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --srg=0.0005 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --srg=0.0005 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --srg=0.0005 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --srg=0.0005 $2
python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --srg=0.0005 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --dueling --srg=0.0005 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --dueling --srg=0.0005 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --dueling --srg=0.0005 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --dueling --srg=0.0005 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --dueling --srg=0.0005 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --dueling --srg=0.0005 $2

# IST 
# # SRG = 0.0008
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --srg=0.0008 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --srg=0.0008 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --dueling --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --dueling --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --dueling --srg=0.0008 $2

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=1 --double --dueling --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=2 --double --dueling --srg=0.0008 $2
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=$1 --nsteps=3 --double --dueling --srg=0.0008 $2

sstat -p -j $SLURM_JOB_ID.batch --format=JobID,MaxRss,MaxVMSize,NTasks,ConsumedEnergy
##sacct -j %j --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

## USAGE
## sbatch ist_single_job.sh <seed> <envname>
## sbatch ist_single_job.sh 1234 mypong

