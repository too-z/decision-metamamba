
# Decision MetaMamba


## Instructions
You need to install the metamamba library first using python.
Use the following command for installation:
```
cd path-to-DMM/mamba
python setup.py install
```
After the installation is complete, you can train each dataset using the run.sh script.

To test the Kitchen environment, it's necessary to ensure that the dm_control package version is compatible with the installed MuJoCo version.
The package versions used in experiments are specified in the requirements.txt.
Additionally, you need to transform the data using the kitchen_to_sparse.py script.


## Acknowledgements
Our Decision MetaMamba code is based on 
[decision-transformer](https://github.com/kzl/decision-transformer)
[decision-convformer](https://github.com/beanie00/Decision-ConvFormer).


