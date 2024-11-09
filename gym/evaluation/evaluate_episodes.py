import os
import sys
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def draw_feature_map(feature_map, name):
    plt.figure(figsize=(14, 6))
    sns.heatmap(feature_map, cmap='viridis', annot=False, vmax=5, vmin=-5)  # `annot=True` shows values in cells
    plt.title(name)
    plt.show()
    
def draw_feature_map_token_mixer(feature_map, name):
    plt.figure(figsize=(14, 6))
    sns.heatmap(feature_map, cmap='summer', annot=False, vmax=5, vmin=-5)  # `annot=True` shows values in cells
    plt.title(name)
    plt.show()


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    state = env.reset()
    sys.stdout = old_stdout
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        max_length = 8
        
        states_ = states
        actions_ = actions
        rewards_ = rewards
        
        states=(states.to(dtype=torch.float32) - state_mean) / state_std
        returns_to_go=target_return.to(dtype=torch.float32)
        
        states = states.reshape(1, -1, state_dim)
        actions = actions.reshape(1, -1, act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)

        states = states[:,-max_length:]
        actions = actions[:,-max_length:]
        returns_to_go = returns_to_go[:,-max_length:]
        
        states = torch.cat(
			[torch.zeros((states.shape[0], max_length-states.shape[1], state_dim), device=states.device), states],
			dim=1).to(dtype=torch.float32)
        actions = torch.cat(
			[torch.zeros((actions.shape[0], max_length - actions.shape[1], act_dim), device=actions.device), actions],
			dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
			[torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
			dim=1).to(dtype=torch.float32)
        
        action, hidden_states_before_ssm, hidden_states_after_ssm, hidden_states_before_token_mixer, hidden_states_after_token_mixer,  = model(states, actions, returns_to_go)
        action = action[0,-1]
        
        states = states_
        actions = actions_
        rewards = rewards_
        
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        
        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break
    
    draw_feature_map_token_mixer(hidden_states_before_token_mixer[0].detach().cpu().numpy(), "hidden_states_before_token_mixer")
    draw_feature_map_token_mixer(hidden_states_after_token_mixer[0].detach().cpu().numpy(), "hidden_states_after_token_mixer")
    
    draw_feature_map(hidden_states_before_ssm[0].detach().cpu().numpy(), "hidden_states_before_ssm")
    draw_feature_map(hidden_states_after_ssm[0].detach().cpu().numpy(), "hidden_states_after_ssm")
    
    return episode_return, episode_length
