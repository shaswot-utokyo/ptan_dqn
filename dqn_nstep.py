# Training DQN using PTAN library

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) 
# suppress numpy future warnings 
# warning created due to issues with tensorflow 1.14


import gym
import ptan
import argparse

import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from lib import dqn_model, common

import datetime
import os
import random
import numpy as np

REWARD_STEPS_DEFAULT = 2

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=False, action="store_true", help="Enable GPU [default:False]")
parser.add_argument("--seed", default=42, type=int, help="Set seed [default: 42]")
parser.add_argument("--nsteps", default=REWARD_STEPS_DEFAULT, type=int, help="Count of steps to unroll Bellman")
parser.add_argument("experiment", help="Experiment to run. Specified in ./lib/common.py")



args = parser.parse_args()

# set device
USE_GPU = args.cuda
USE_CUDA = torch.cuda.is_available() and USE_GPU
device = torch.device("cuda" if USE_CUDA else "cpu")


# set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

experiment = args.experiment
params = common.HYPERPARAMS[experiment]

print("SEED: ",seed)
print("EXPERIMENT: ",experiment)
print("TIME: ", datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
print("\n")

env = gym.make(params['env_name'])
env = ptan.common.wrappers.wrap_dqn(env)

tag = params['run_name'] + "-nsteps_" + str(args.nsteps) +  '_'+ str(seed)
writer_folder = './runs/'+ params['run_name'] + "-nsteps_" + str(args.nsteps) + "/" + str(seed) +  '_' + datetime.datetime.now().strftime("%d-%b-%H-%M-%S")
writer = SummaryWriter(log_dir=writer_folder)

net = dqn_model.DQN(env.observation_space.shape, 
                    env.action_space.n).to(device)

tgt_net = ptan.agent.TargetNet(net)

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
epsilon_tracker = common.EpsilonTracker(selector, params)
# EpsilonTracker has only one method frame(frame_idx) which changes the epsilon value 
# of selector=>ptan.actions.EpsilonGreedyActionSelector depending upon the frame

agent = ptan.agent.DQNAgent(net, selector, device=device)

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=args.nsteps)

buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])

optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

frame_idx = 0
with common.RewardTracker(writer, params['stop_reward']) as reward_tracker: #create a reward tracker object
    while True:
        frame_idx += 1
        # ExperienceReplayBuffer asks the ExperienceSourceFirstLast to iterate by one step to get the next transition
        # ExperienceSourceFirstLast feeds observation to obtain action
        # Agent calculated Q-values through the NN
        # Action selector selects action
        # Action is fed into ExperienceSource to obtain reward and next obs
        # Buffer stores transition in FIFO order
        buffer.populate(1) # iterates ExperienceReplayBuffer by 1 step.
                            # this in turn iterates exp_source [ExperienceSourceFirstLast] by one step
                            # one single experience step
                            # Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
                            
                            # Class ExperienceSource provides us full subtrajectories of given length as the list of (s, a, r, s') objects.
                            # Now it returns single object on every iteration, which is again a namedtuple with the following fields:

                            # state: state which we used to decide on action to make TYPE: numpy
                            # action: action we've done at this step
                            # reward: partial accumulated reward for steps_count (in our case, steps_count=1, so it is equal to immediate reward)
                            # last_state: the state we've got after executing the action. If our episode ends, we have None here

                            # For every trajectory piece it calculates discounted reward and emits only first and last states and action taken in the first state.
        epsilon_tracker.frame(frame_idx)
        

        new_rewards = exp_source.pop_total_rewards() # get rewards from the episodes
        if new_rewards:
            if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                break

        if len(buffer) < params['replay_initial']:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma']**args.nsteps, device=device)
        if frame_idx % 1E3 == 0:
            writer.add_scalar("loss", loss_v, frame_idx)
        loss_v.backward()
        optimizer.step()

        if frame_idx % params['target_net_sync'] == 0:
            tgt_net.sync()
            
        if frame_idx > 3E6:
            break

import os.path
cur_folder = os.getcwd()
model_folder = os.path.join(cur_folder,"models")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_file = os.path.join(model_folder, (tag + ".pt"))
torch.save(net.state_dict(), model_file)