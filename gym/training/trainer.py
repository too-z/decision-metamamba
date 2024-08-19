import numpy as np
import torch

import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        
        for i in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            if (i+1) % 1000 == 0:
                print(f'Step {i+1}', end=' | ', flush=True)

        logs['time/training'] = time.time() - train_start
        
        eval_start = time.time()
        self.model.eval()
        
        max_score = 0
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v
                if v > max_score:
                    max_score = v
        logs['evaluation/target_max_d4rl_score'] = max_score
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('\n' + ('=' * 120))
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
        print('=' * 120)
        return logs
