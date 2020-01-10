#!/bin/bash
#PBS -q h-regular
#PBS -l select=1:mpiprocs=1:ompthreads=6
#PBS -W group_list=gk37
#PBS -l walltime=15:00:00

cd $PBS_O_WORKDIR
./etc/profile.d/modules.sh

module load anaconda3/2019.03 cuda9/9.2.148
export PYTHONUSERBASE=/lustre/gk37/k37004/envs/rl
export PATH=$PYTHONUSERBASE/bin:$PATH


python dqn_basic.py --cuda --seed=${seed} ${experiment}

