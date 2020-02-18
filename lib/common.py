import sys
import time
import numpy as np
import torch
import torch.nn as nn


HYPERPARAMS = {
    'beamrider-v0': {
        'env_name':         "BeamRiderNoFrameskip-v4",
        'stop_reward':       6000,
        'run_name':         'beamrider-v0',
        'replay_size':      100000, 
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },
     'pong3': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong_b128',
        'replay_size':      100000, 
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       128 # increased batch size to 128
    },
    'pong2': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong_b64',
        'replay_size':      100000, 
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64 # increased batch size to 64
    },
     'pong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },
    'mypong3': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'mypong_b128',
        'replay_size':      10000, #decreased from original by a factor of 10
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       128 # increased batch size to 128 
    },
    'mypong2': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'mypong_b64',
        'replay_size':      10000, #decreased from original by a factor of 10
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64 # increased batch size to 64
    },
    'mypong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'mypong',
        'replay_size':      10000, #decreased from original by a factor of 10
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },
    'breakout-small': {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    },
    'mybreakout': {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small_rb',
        'replay_size':      10 ** 5,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    },
    'breakout': {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    },
    'invaders': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'invaders',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
}

# This function is call sequence is:
# batch = buffer.sample(BATCH_SIZE) where buffer is a ptan.experience.ExperienceReplayBuffer().
# ptan.experience.ExperienceReplayBuffer() is an iterable

# returns a list of lists. Each sublist contains a namedtuple 
# Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])

# calc_loss(batch) calls unpack_batch(batch)
# batch may contain experience of any datatypes
"""def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], [] 
    for exp in batch:
        state = np.array(exp.state, copy=False) #copy=False ensures that new memory is not allocated
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway with code next_state_values[done_mask] = 0.0

        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)
"""
def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], [] 
    for exp in batch:
        state = np.array(exp.state, copy=False) #copy=False ensures that new memory is not allocated
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway with code next_state_values[done_mask] = 0.0

        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.bool_), np.array(last_states, copy=False)

def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    # np.array_split splits the array into sub-arrays 
    # (in this case, each sub-array has a max size of 64)
    # For an array of length l that should be split into n sections, 
    # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
    for batch in np.array_split(states, 64): 
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v) #get q_values of all actions for all the 64 states
        best_action_values_v = action_values_v.max(1)[0] # get maximum q_values for each of the 64 states
        mean_vals.append(best_action_values_v.mean().item()) # take mean of the 64 max q_values
    return np.mean(mean_vals)

def calc_values_of_states_srg(states, net, device="cpu"):
    mean_vals = []
    # np.array_split splits the array into sub-arrays 
    # (in this case, each sub-array has a max size of 64)
    # For an array of length l that should be split into n sections, 
    # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
    for batch in np.array_split(states, 64): 
        states_v = torch.tensor(batch).to(device)
        action_values_v, _ = net(states_v) #get q_values of all actions for all the 64 states
        best_action_values_v = action_values_v.max(1)[0] # get maximum q_values for each of the 64 states
        mean_vals.append(best_action_values_v.mean().item()) # take mean of the 64 max q_values
    return np.mean(mean_vals)

def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu", double=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch) # returns numpy arrays

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones,dtype=torch.bool).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double: # double-DQN
        next_state_actions = net(next_states_v).max(1)[1] # get greedy actions from policy net
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_loss_srg_ratio(batch, net, tgt_net, gamma, device="cpu", double=False, loss_ratio=1E-4):
    states, actions, rewards, dones, next_states = unpack_batch(batch) # returns numpy arrays

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones,dtype=torch.bool).to(device)

    state_action_all_values, state_features = net(states_v)
    state_action_values = state_action_all_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        _, next_state_features = net(next_states_v)
    next_state_features.detach()
    
    next_state_all_values, _ = tgt_net(next_states_v)
    next_state_values = next_state_all_values.max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    
    if double: # double-DQN
        next_state_all_values_net, _ = net(next_states_v)
        next_state_actions = next_state_all_values_net.max(1)[1] # get greedy actions from policy net
        next_state_all_values_tgt, _ = tgt_net(next_states_v)
        next_state_values = next_state_all_values_tgt.gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_all_values, _ = tgt_net(next_states_v)
        next_state_values = next_state_all_values.max(1)[0]
    
    next_state_values[done_mask] = 0.0
    expected_state_action_values = next_state_values * gamma + rewards_v
    
    feature_loss = torch.dist(state_features, next_state_features, p=2)
    qvalue_loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    
    loss = loss_ratio*feature_loss + qvalue_loss
    
    return loss, feature_loss, qvalue_loss

# def calc_loss_srg(batch, net, tgt_net, gamma, device="cpu", double=False):
#     states, actions, rewards, dones, next_states = unpack_batch(batch) # returns numpy arrays

#     states_v = torch.tensor(states).to(device)
#     next_states_v = torch.tensor(next_states).to(device)
#     actions_v = torch.tensor(actions).to(device)
#     rewards_v = torch.tensor(rewards).to(device)
#     done_mask = torch.tensor(dones,dtype=torch.bool).to(device)

#     state_action_all_values, state_features = net(states_v)
#     state_action_values = state_action_all_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
#     with torch.no_grad():
#         _, next_state_features = net(next_states_v)
#     next_state_features.detach()
    
#     next_state_all_values, _ = tgt_net(next_states_v)
#     next_state_values = next_state_all_values.max(1)[0]
#     next_state_values[done_mask] = 0.0
#     next_state_values = next_state_values.detach()

    
#     if double: # double-DQN
#         next_state_all_values_net, _ = net(next_states_v)
#         next_state_actions = next_state_all_values_net.max(1)[1] # get greedy actions from policy net
#         next_state_all_values_tgt, _ = tgt_net(next_states_v)
#         next_state_values = next_state_all_values_tgt.gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
#     else:
#         next_state_all_values, _ = tgt_net(next_states_v)
#         next_state_values = next_state_all_values.max(1)[0]
    
#     next_state_values[done_mask] = 0.0
#     expected_state_action_values = next_state_values * gamma + rewards_v
    
#     feature_loss = torch.dist(state_features, next_state_features, p=2)
#     qvalue_loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    
#     loss = 1E-4*feature_loss + qvalue_loss
    
#     return loss, feature_loss, qvalue_loss

# def calc_loss_srg_disabled(batch, net, tgt_net, gamma, device="cpu"):
#     states, actions, rewards, dones, next_states = unpack_batch(batch) # returns numpy arrays

#     states_v = torch.tensor(states).to(device)
#     next_states_v = torch.tensor(next_states).to(device)
#     actions_v = torch.tensor(actions).to(device)
#     rewards_v = torch.tensor(rewards).to(device)
#     done_mask = torch.tensor(dones,dtype=torch.bool).to(device)

#     state_action_all_values, state_features = net(states_v)
#     state_action_values = state_action_all_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
#     with torch.no_grad():
#         _, next_state_features = net(next_states_v)
#     next_state_features.detach()
    
#     next_state_all_values, _ = tgt_net(next_states_v)
#     next_state_values = next_state_all_values.max(1)[0]
#     next_state_values[done_mask] = 0.0
#     next_state_values = next_state_values.detach()

#     expected_state_action_values = next_state_values * gamma + rewards_v
    
#     feature_loss = torch.dist(state_features, next_state_features, p=2)
#     qvalue_loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    
#     loss = qvalue_loss
    
#     return loss, feature_loss, qvalue_loss



# tracks the total reward at the end of every episode
# tracks the mean reward for the last 100 episodes
# write the values in tensorboard
# check if the mean reward exceeds the threshold and if the game has been solved
class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
#         print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
#            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
#         ))
#         sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr
