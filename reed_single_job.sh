#!/bin/bash
#PBS -q h-regular
#PBS -l select=1:mpiprocs=1:ompthreads=12
#PBS -W group_list=gk37
#PBS -l walltime=15:00:00

cd $PBS_O_WORKDIR
./etc/profile.d/modules.sh

module load anaconda3/2019.03 cuda9/9.2.148
export PYTHONUSERBASE=/lustre/gk37/k37004/envs/rl
export PATH=$PYTHONUSERBASE/bin:$PATH

# OLD JOBS
#################################################################################################
# python dqn_basic.py --cuda --seed=${seed} ${experiment}
# python dqn_srg.py --cuda --seed=${seed} ${experiment}
# python dqn_double.py --cuda --seed=${seed} --nsteps=${nsteps} --double ${experiment}
#################################################################################################

# NEW JOBS
# USAGE:
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=<seed_value> --nsteps=<rollout_length> --double --dueling --srg=<srg_ratio> <experiment>

# # SRG = OFF
# REED
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 ${experiment}

# IST
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double ${experiment}

# REED 
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --dueling ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --dueling ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --dueling ${experiment}

# IST
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --dueling ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --dueling ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --dueling ${experiment}

# IST
# # SRG = 0.0001
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --srg=0.0001 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --srg=0.0001 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --dueling --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --dueling --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --dueling --srg=0.0001 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --dueling --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --dueling --srg=0.0001 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --dueling --srg=0.0001 ${experiment}


# REED
# # SRG = 0.0002
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --srg=0.0002 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --srg=0.0002 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --dueling --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --dueling --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --dueling --srg=0.0002 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --dueling --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --dueling --srg=0.0002 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --dueling --srg=0.0002 ${experiment}

# XXXIST
# # SRG = 0.0005
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --srg=0.0005 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --srg=0.0005 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --dueling --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --dueling --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --dueling --srg=0.0005 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --dueling --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --dueling --srg=0.0005 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --dueling --srg=0.0005 ${experiment}

# # IST = 0.0008
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --srg=0.0008 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --srg=0.0008 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --dueling --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --dueling --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --dueling --srg=0.0008 ${experiment}

# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=1 --double --dueling --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=2 --double --dueling --srg=0.0008 ${experiment}
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=${seed} --nsteps=3 --double --dueling --srg=0.0008 ${experiment}
