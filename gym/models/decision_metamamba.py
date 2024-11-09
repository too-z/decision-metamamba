import math
import torch
import torch.nn as nn

import transformers

from transformers.utils import logging

from functools import partial

from mamba_ssm.modules.mamba_simple import Mamba

try:
	from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except:
	RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

logger = logging.get_logger(__name__)


def create_block(
	d_model,
	ssm_cfg=None,
	norm_epsilon=1e-5,
	rms_norm=False,
	layer_idx=None,
	token_mixer='conv',
):
	if ssm_cfg is None:
		ssm_cfg = {}

	block = Mamba(
		d_model,
		layer_idx=layer_idx,
		token_mixer=token_mixer,
		**ssm_cfg
	)
	# block.layer_idx = layer_idx
	return block


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
	def __init__(
		self,
		config,
		ssm_cfg=None,
		norm_epsilon: float = 1e-5,
		rms_norm: bool = False,
		initializer_cfg=None,
	) -> None:
		super().__init__()

		self.drop = nn.Dropout(config.embd_pdrop)
		self.layers = nn.ModuleList(
			[
				create_block(
					config.n_embd,
					ssm_cfg=ssm_cfg,
					norm_epsilon=norm_epsilon,
					rms_norm=rms_norm,
					layer_idx=i,
					token_mixer=config.token_mixer
				)
				for i in range(config.n_layer)
			]
		)
		
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
		for layer in self.layers:
			hidden_states = layer(
				hidden_states, inference_params
			)
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
		
	def forward(self, states, actions, returns_to_go):
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
		x = self.model(hidden_states=stacked_inputs)

		# reshape x so that the second dimension corresponds to the original
		# returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
		x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(0, 2, 1, 3)

		state_reps = x[:,1]
		
		action_preds = self.predict_action(state_reps)  # predict next action given state

		return action_preds
