import torch
import numpy as np
import gym

from models.decision_metamamba import DecisionMetaMamba
from training.dmm_trainer import DecisionMetaMambaTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def get_env_info(env_name, dataset):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 7200, 36000, 72000]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 24000, 120000, 240000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 10000, 50000, 100000]
        scale = 1000.
    elif env_name == 'antmaze':
        import d4rl
        env = gym.make(f'{env_name}-{dataset}-v2')
        max_ep_len = 1000
        env_targets = [1.0, 10.0, 100.0, 1000.0, 100000.0] # successful trajectories have returns of 1, unsuccessful have returns of 0
        scale = 1.
    elif 'kitchen' in env_name:
        import d4rl
        env = gym.make(f'kitchen-{dataset}-v0')
        max_ep_len = 1000
        env_targets = [4.0, 10.0, 100.0] # successful trajectories have returns of 1, unsuccessful have returns of 0
        scale = 1.
    else:
        raise NotImplementedError
    
    return env, max_ep_len, env_targets, scale

def get_model_optimizer(variant, state_dim, act_dim, K, max_ep_len, device):
    
    model = DecisionMetaMamba(
        env_name=variant['env'],
        dataset=variant['dataset'],
        device=device,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        drop_p=variant['dropout'],
        window_size=variant['window_size'],
        activation_function=variant['activation_function'],
        resid_pdrop=variant['dropout'],
        global_sequence_mixer=variant['global_sequence_mixer'],
    )
    model = model.to(device=device)
    
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    return model, optimizer, scheduler

def get_trainer(**kwargs):
    return DecisionMetaMambaTrainer(**kwargs)
