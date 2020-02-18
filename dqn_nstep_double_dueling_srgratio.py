# Training DQN using PTAN library
# USAGE:
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=<seed_value> --nsteps=<rollout_length> --double --dueling --srg=<srg_ratio> <experiment>
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=123 --nsteps=3 --double --dueling pong
# python dqn_nstep_double_dueling_srgratio.py --cuda --seed=123 --nsteps=3 --double --dueling --srg=0.0001 pong

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

SEED_DEFAULT = 42
ROLLOUT_STEPS_DEFAULT = 1
# SRG_RATIO_DEFAULT = 1E-4
# evaluate Q-values of random states
NO_OF_STATES_TO_EVALUATE = 1000 # how many states to sample to evaluate mean q-value
EVAL_FREQ = 100 # how often to evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=False, action="store_true", help="Enable GPU [default:False]")
parser.add_argument("--seed", default=SEED_DEFAULT, type=int, help="Set seed [default: 42]")
parser.add_argument("--nsteps", default=ROLLOUT_STEPS_DEFAULT, type=int, help="Count of steps to unroll Bellman")
parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
parser.add_argument("--dueling", default=False, action="store_true", help="Enable double DQN")
parser.add_argument("--srg", default=None, type=float, help="Enable state-feature regularization")
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
print("TIME: ", datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
print("\n")

env = gym.make(params['env_name'])
env = ptan.common.wrappers.wrap_dqn(env)

# Tensorboard log file/directory
nstep_tag = str(args.nsteps)+"step"
tag = params['run_name'] + '-' + nstep_tag  # nstep rollouts
if args.double:
    double_tag = 'double'
    tag = tag + '-' + double_tag # double dqn
    
if args.dueling:
    dueling_tag = 'dueling'
    tag = tag + '-' + dueling_tag # double dqn

if args.srg is not None:
    srg_tag = 'srg_'+ str(args.srg)
    tag = tag + '-'+ srg_tag  
tag = tag + '-' + str(seed) # seed
tag = tag + '-' + datetime.datetime.now().strftime("%d-%b-%H-%M-%S")
print("TAG: ",tag)
writer_folder = './runs/'+ params['run_name'] + "-" + nstep_tag +"/" + tag
writer = SummaryWriter(log_dir=writer_folder)

if args.srg is not None:
    if args.dueling:
        net = dqn_model.DuelingDQN_srg(env.observation_space.shape, 
                        env.action_space.n).to(device)
    else:
        net = dqn_model.DQN_srg(env.observation_space.shape, 
                            env.action_space.n).to(device)
else:
    if args.dueling:
        net = dqn_model.DuelingDQN(env.observation_space.shape, 
                        env.action_space.n).to(device)
    else:
        net = dqn_model.DQN(env.observation_space.shape, 
                            env.action_space.n).to(device)

tgt_net = ptan.agent.TargetNet(net)

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
epsilon_tracker = common.EpsilonTracker(selector, params)
# EpsilonTracker has only one method frame(frame_idx) which changes the epsilon value 
# of selector=>ptan.actions.EpsilonGreedyActionSelector depending upon the frame

class DQNAgent_srg(ptan.agent.BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=ptan.agent.default_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_v, _ = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states
    
if args.srg is not None:
    agent = DQNAgent_srg(net, selector, device=device)
else:
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=args.nsteps)

buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])

optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

frame_idx = 0
eval_states = None # will be populated with held-out states

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

        # evaluate states
        if eval_states is None:
                eval_states = buffer.sample(NO_OF_STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        if args.srg is not None: #state regularization
            loss_v, feature_loss, qvalue_loss = common.calc_loss_srg_ratio(batch, net, tgt_net.target_model,
                                                                           gamma=params['gamma']**args.nsteps, 
                                                                           device=device, 
                                                                           double=args.double,
                                                                           loss_ratio=args.srg)
            if frame_idx % 1E3 == 0:
                writer.add_scalar("loss", loss_v, frame_idx)
                writer.add_scalar("feature_loss", feature_loss, frame_idx)
                writer.add_scalar("qvalue_loss", qvalue_loss, frame_idx)
        else:
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, 
                                          gamma=params['gamma']**args.nsteps,
                                          device=device,
                                          double=args.double)
            if frame_idx % 1E3 == 0:
                writer.add_scalar("loss", loss_v, frame_idx)
        loss_v.backward()
        optimizer.step()

        if frame_idx % params['target_net_sync'] == 0:
            tgt_net.sync()
        
        if frame_idx % EVAL_FREQ == 0:
            if args.srg is not None:
                mean_val = common.calc_values_of_states_srg(eval_states, net, device=device)
            else:
                mean_val = common.calc_values_of_states(eval_states, net, device=device)
            writer.add_scalar("values_mean", mean_val, frame_idx)        
            
        if frame_idx > 1E6:
            break

import os.path
cur_folder = os.getcwd()
model_folder = os.path.join(cur_folder,"models")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_file = os.path.join(model_folder, (tag + ".pt"))
torch.save(net.state_dict(), model_file)
