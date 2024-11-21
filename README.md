
# Decision Metamamba

## Overview

Decision MetaMamba Architecture:
![image info](./architecture.png)

A link of the paper can be found on [arXiv](https://arxiv.org/abs/2408.10517).


## Instructions
You need to install the metamamba library first using pip.
Use the following command for installation:
```
cd path-to-DMM/metamamba
pip install -e .
```
After the installation is complete, you can train each dataset using the run.sh script.

## Feature Map Activation for Proximal and Distal steps

Hidden states before and after a Selective Scan SSM
![hidden state before ssm](./hidden_state_before_ssm.png)
![hidden state after ssm](./hidden_state_after_ssm.png)

Hidden states before and after a token mixer
![hidden state before token mixer](./hidden_state_before_token_mixer.png)
![hidden state after token mixer](./hidden_state_after_token_mixer.png)

## Acknowledgements
Our Decision Metamamba code is based on 
[decision-transformer](https://github.com/kzl/decision-transformer)
[decision-convformer](https://github.com/beanie00/Decision-ConvFormer).

and [mamba](https://github.com/state-spaces/mamba)


## Citation

Please cite out paper as:
```
@misc{kim2024integratingmultimodalinputtoken,
      title={Integrating Multi-Modal Input Token Mixer Into Mamba-Based Decision Models: Decision MetaMamba}, 
      author={Wall Kim},
      year={2024},
      eprint={2408.10517},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.10517}, 
}
```

## License

MIT
