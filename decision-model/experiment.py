import gym
import torch
import numpy as np
from d4rl import infos

import argparse
import pickle
import random
import sys
from datetime import datetime
import os

from evaluation.evaluate_episodes import evaluate_episode_rtg
from utils import discount_cumsum, get_env_info, get_model_optimizer, get_trainer

def experiment(variant):
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']

    env, max_ep_len, env_targets, scale = get_env_info(env_name, dataset)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    if dataset == 'medium-expert':
        dataset_path = f'data/{env_name}-expert-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        dataset_path = f'data/{env_name}-medium-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories += pickle.load(f)
        random.shuffle(trajectories)
    elif 'kitchen' in env_name:
        dataset_path = f'data/{env_name}-v0-sparse.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    
    num_timesteps = sum(traj_lens)

    print('=' * 120)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')

    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 120)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    prefix = variant['env'] + "_" + variant['dataset']
    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    

    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training
            
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, rtg, mask = [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, rtg, mask
        

    def eval_episodes(target):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device
                    )
                returns.append(ret)
                lengths.append(length)
            if 'kitchen' in env_name:
                reward_min = infos.REF_MIN_SCORE[f"kitchen-{dataset}-v0"]
                reward_max = infos.REF_MAX_SCORE[f"kitchen-{dataset}-v0"]
            else:
                reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v2"]
                reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v2"]
            return {
                f'target_{target}_d4rl_score': (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),
            }
        return fn

    model, optimizer, scheduler = get_model_optimizer(variant, state_dim, act_dim, K, max_ep_len, device)
    print(f"# of parameters = {sum(p.numel() for p in model.parameters())}")
    loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)

    trainer = get_trainer(
        model=model,
        batch_size=batch_size,
        get_batch=get_batch,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    max_score = 0    
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if outputs['evaluation/target_max_d4rl_score'] > max_score:
            max_score = outputs['evaluation/target_max_d4rl_score']
    print('evaluation/total_max_d4rl_score:', max_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse, no-reward-decay
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--window_size', type=int, default=6)
    parser.add_argument('--global_sequence_mixer', type=str, default='mamba1') # mamba2

    args = parser.parse_args()

    experiment(variant=vars(args))
