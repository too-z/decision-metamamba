import math
import torch
import torch.nn as nn

import transformers

from transformers.utils import logging

from functools import partial

from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2
from mamba_ssm.modules.mamba2 import Mamba2



logger = logging.get_logger(__name__)


class DenseSequenceMixer(nn.Module):
    def __init__(self, window_size, hidden_size):
        super().__init__()
        self.window_size = window_size
        self.lin = nn.Linear(hidden_size*window_size, hidden_size, bias=False)

    def forward(self, x): # input shape like [batch_size, sequence_length, d_model]
        padded_tensor = torch.nn.functional.pad(x, (0, 0, self.window_size - 1, 0)) # (64, 24, 128) to (64, 29, 128)
        padded_tensor = torch.flatten(padded_tensor.unfold(1,self.window_size,1).transpose(2,3), start_dim=2) # flatten with window sized vectors
        lin_tensor = self.lin(padded_tensor)
        return lin_tensor


def _init_weights(
    module,
    initializer_range=0.02
):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if isinstance(module, (nn.Linear)) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Model(nn.Module):
	def __init__(self, config, initializer_cfg=None) -> None:
		super().__init__()

		self.drop = nn.Dropout(config.embd_pdrop)
  
		self.ln_1 = nn.LayerNorm(config.n_embd)
		
		if config.env_name == 'antmaze' or 'kitchen' in config.env_name:
			self.local_sequence_mixers = [DenseSequenceMixer(window_size=config.window_size, hidden_size=config.n_embd).to(device=config.device)] * config.n_layer
		else:
			self.local_sequence_mixers = nn.ModuleList([DenseSequenceMixer(window_size=config.window_size, hidden_size=config.n_embd) for _ in range(config.n_layer)])
		
		self.ln_2 = nn.LayerNorm(config.n_embd)
        
		if config.global_sequence_mixer == 'mamba1':
			self.global_sequence_mixers = nn.ModuleList([Mamba(d_model=config.n_embd, expand=1, layer_idx=i) for i in range(config.n_layer)])
		elif config.global_sequence_mixer == 'mamba2':
			self.global_sequence_mixers = nn.ModuleList([Mamba2(d_model=config.n_embd, expand=1, headdim=16, use_mem_eff_path=False, layer_idx=i) for i in range(config.n_layer)])
		else:
			raise "global sequence mixer not implemented"
		
		self.apply(
			partial(
				_init_weights,
				**(initializer_cfg if initializer_cfg is not None else {}),
			)
		)
		self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
	
	def forward(
		self,
		hidden_states=None,
		inference_params=None,
	):
		hidden_states = self.drop(hidden_states)
		
		for local_sequence_mixer, global_sequence_mixer in zip(self.local_sequence_mixers, self.global_sequence_mixers):
			_hidden_states = self.ln_1(hidden_states)
			_hidden_states = local_sequence_mixer(_hidden_states)
			
			hidden_states = hidden_states + _hidden_states
			
			_hidden_states = self.ln_2(hidden_states)
			_hidden_states = global_sequence_mixer(_hidden_states, inference_params)
			
			hidden_states = hidden_states + _hidden_states
			
		hidden_states = self.ln_f(hidden_states)
		return hidden_states


class DecisionMetaMamba(nn.Module):
	def __init__(
		self,
		state_dim,
		act_dim,
		hidden_size,
		max_length=None,
		action_tanh=True,
		**kwargs
	):
		super().__init__()

		self.state_dim = state_dim
		self.act_dim = act_dim
		self.max_length = max_length
		self.hidden_size = hidden_size

		config = transformers.GPT2Config(
			n_embd=hidden_size,
			**kwargs
		)
		self.env_name = config.env_name

		self.model = Model(config)

		self.embed_return = torch.nn.Linear(1, hidden_size)
		self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
		self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

		self.embed_ln = nn.LayerNorm(hidden_size)

		self.predict_action = nn.Sequential(
			*([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
		)
		
	def forward(self, states, actions, returns_to_go, inference_params=None):
		batch_size, seq_length = states.shape[0], states.shape[1]
		
		state_embeddings = self.embed_state(states)
		returns_embeddings = self.embed_return(returns_to_go)
		action_embeddings = self.embed_action(actions)

		# this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
		# which works nice in an autoregressive sense since states predict actions
		num_token_type = 3
		stacked_inputs = torch.stack(
			(returns_embeddings, state_embeddings, action_embeddings), dim=1
		).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
		stacked_inputs = self.embed_ln(stacked_inputs)

		# we feed in the input embeddings (not word indices as in NLP) to the model
		x = self.model(hidden_states=stacked_inputs, inference_params=inference_params)

		# reshape x so that the second dimension corresponds to the original
		# returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
		x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(0, 2, 1, 3)

		state_reps = x[:,1]
		
		action_preds = self.predict_action(state_reps)  # predict next action given state

		return action_preds

	def get_action(self, states, actions, returns_to_go, inference_params):
		# we don't care about the past rewards in this model

		states = states.reshape(1, -1, self.state_dim)
		actions = actions.reshape(1, -1, self.act_dim)
		returns_to_go = returns_to_go.reshape(1, -1, 1)

		states = states[:,-self.max_length:]
		actions = actions[:,-self.max_length:]
		returns_to_go = returns_to_go[:,-self.max_length:]

		states = torch.cat([torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],dim=1).to(dtype=torch.float32)
		actions = torch.cat([torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],dim=1).to(dtype=torch.float32)
		returns_to_go = torch.cat([torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],dim=1).to(dtype=torch.float32)
		
		action_preds = self.forward(states, actions, returns_to_go, inference_params)

		return action_preds[0,-1]
