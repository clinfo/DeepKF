# Deep kalman filter / deep state space models

Tensor flow implementation of the deep state space models

- Krishnan, Rahul G., Uri Shalit, and David Sontag. "Deep kalman filters." arXiv preprint arXiv:1511.05121 (2015).
- Krishnan, Rahul G., Uri Shalit, and David Sontag. "Structured Inference Networks for Nonlinear State Space Models", In AAAI 2017

[README: Japanese](./README_jp.md)
## citation
This software is developped for a part of the following study:
```
@misc{nakamura2023new,
      title={A New Deep State-Space Analysis Framework for Patient Latent State Estimation and Classification from EHR Time Series Data}, 
      author={Aya Nakamura and Ryosuke Kojima and Yuji Okamoto and Eiichiro Uchino and Yohei Mineharu and Yohei Harada and Mayumi Kamada and Manabu Muto and Motoko Yanagita and Yasushi Okuno},
      year={2023},
      eprint={2307.11487},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
### Requirements
* python3 (> 3.3)
  * tensorflow (>0.12)
  * joblib

### Anaconda install
First, please install anaconda by the official anaconda instruction [https://conda.io/docs/user-guide/install/linux.html].

#### Installation Reference

- Installing pyenv
```
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

- Found latest version of anaconda
```
pyenv install -l | grep ana
```
（anaconda3-2019.10 is recommended）
- Installing anaconda
```
pyenv install anaconda3-2019.10
pyenv rehash
pyenv global anaconda3-2019.10
echo 'export PATH="$PYENV_ROOT/versions/anaconda3-2019.10/bin/:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda update conda
```

Next, please install following libraries if you have GPU:
```
pip install --ignore-installed --upgrade tensorflow_gpu==1.15.0
pip install joblib
```
if you use only CPUs:
```
pip install --ignore-installed --upgrade tensorflow==1.15.0
pip install joblib
```
## How to run the sample

You can use the following command to run the sample:

```
$sh run_sample.sh
```

## About the sample data
The sample data is as follows:

```
sample/sample.csv
```

From this csv file, the following script is used to create the data for training.
```
python build_sample.py
```

The above command will extract the data under the ``data/`` directory.
```
data/sample_*
```



## About the sample scripts

### Training

```
dmm train --config sample/config.json --hyperparam sample/hyparam.json 
```

`config.json` is a configuration file that describes settings, for example, input/output files.
`hyparam.json` is a configuration file that mainly describes settings related to learning.


### Prediction

```
dmm test --config sample/config.json --hyperparam model/hyparam.result.json --save-config ./model/config.result.json
```

`model/hyparam.result.json` is a configuration file that includes parameters automatically determined from the `hyparam.json` file given during training phase.


### Filtering

```
dmm filter --config model/config.result.json --hyperparam model/hyparam.result.json
```


### Estimation of states

```
dmm field --config model/config.result.json --hyperparam model/hyparam.result.json
```



### Execution of the procedure with one command

```
dmm train,test,filter,field --config sample/config.json --hyperparam sample/hyparam.json
```
### Visualization

#### Visualization of training
```
dmm-plot train --config model/config.result.json --obs_dim 0
```

#### Visualization of results of test
```
dmm-plot infer --config model/config.result.json --obs_dim 0
```

This command creates a two-tiered plot.
The upper plot is the estimated state space and the lower plot is the observed space, with only the first dimension of the observed space displayed.
In this example, only the 0th dimension is output by default, but you can specify how many dimensions to output with the ``--obs_dim`` option.
For each entry in the observation space, ``x`` represents the observed value, ``recons`` the reconstructed observed value, and ``pred`` the predicted value.


### Visualization of filtering

```
dmm-plot filter --config model/config.result.json --num_dim 2 --obs_dim 0 all
```

This command creates a three-tiered plot.
The top row is the sampled state space, the middle row is the observed space, and the bottom row plots the predicted vs. actual deviation of the observed space

The state space plots only the first two dimensions.
The ``--num_dim`` option can be used to specify how many dimensions are output.
Each dimension is displayed as dim0, dim1, dim2, ... ....

In addition, the middle row shows only one dimension of the observation space.
In this example, only the 0th dimension is output by default.
The ``x`` is the observed value, ``pred`` is the predicted value (particle), ``pred(mean)`` is the mean of the predicted values (particles), and so on.

The lower row is the same as the observation space and displays the error from the observed values.

### visualization of state space

```
dmm-field-plot --config model/config.result.json
```


This command displays the transition direction of the state space with time.


### Comparison with other dimensionality reduction methods

The following command performs dimensionality reduction by PCA(principal component analysis)

```
dmm-map pca --config model/config.result.json
```

(input file is specified with `--input`)
```
dmm-plot pca --config model/config.result.json --input pca.jbl
```

Other methods such as umap and tsne can be performed by changing the pca part


## Parameters

You can set various parameters in a format like `sample/config.json` and `sample/hyparam.json`.
These are at runtime as:
```
--config sample/config.result.json --hyperparam sample/hyparam.result.json
```

`config.json` contains parameters that do not change over multiple experiments, while `hyperparam.json` contains parameters that change over multiple experiments.
When tuning parameters to observe changes in accuracy, it is basically assumed that only the parameters in hyparam.json will be changed, and config.json will be used for other purposes. If different values are set for the same parameter in hyparam.json and config.json, hyparam.json takes precedence.

### Configuration 

#### *"data_train_npy"/"data_test_npy"*
- This items specifies numpy files for training data/test data
- Time-series multidimensional array data of (number of samples) x (time steps) x (#features)

#### *"mask_train_npy"/"mask_test_npy"*
- These items specifies numpy files for missing/missing training/test data
- 0 means missing, 1 means no missing
- Time-series multidimensional array data of number of data x time x features
- If omitted, all data are assumed to be missing

#### *"steps_train_npy"/"steps_test_npy"*
- These items specifies numpy files that stores the number of valid time steps for training/test data
- Vector of number of data samples
- If omitted, all times are valid
- This options is to handle variable length data where the length of time differs for each data

#### *"data_info_json"*
Information about each data sample

#### *"batch_size"*
- Batch size

#### *"dim"*
- Dimension of the state space
- Smaller sizes do not adequately represent the observed data, resulting in less accurate reconstructions and poor learning.
- If it is large, it is difficult to train and interpret the results.


#### *"epoch"*
- Maximum number of iterations

#### *"patience"*
- Early stopping parameter
- If the validation loss does not decrease more than this number of times, learning stops there.


#### *"load_model"*
- Load and use the saved model (only for testing)


#### *"result_path"*
- "result_path" Saves all the following results to the specified directory
  - "save_model"
  - "save_model_path"
  - "save_result_filter"
  - "save_result_test"
  - "save_result_train"
  - "simulation_path"
  - "evaluation_output"
  - "load_model"
  - "plot_path"
  - "log"

#### *"save_model_path"*
- Where to save the final learn model

#### "save_result_train"/"save_result_test"
- Destination to save the results of training and testing


#### *"train_test_ratio"*
- Specify the ratio of training to validation. When training is 80% and validation is the rest, specify like ``[0.8,0.2]``.

#### *"alpha"*
- Time direction loss weights
- When 0, the time direction loss is always zero.
- If the latent space cannot be learned well, set a smaller value.

#### *"beta"*
- Potential loss weights
- When 0, potential loss is always 0
- If the prediction does not learn well along the potential, set a higher value


#### *"gamma"*
- Potential loss weights
- When 0, potential loss is always 0
- If the prediction does not learn well along the potential, set a higher value

#### *"evaluation_output"*
- json file to output configuration and evaluation values
- The key "evaluation" is added in the same format as config.json, and the evaluation results are stored.


#### *"hyperparameter_input"*
- Hyperparameter file "hyparam.json" can also be specified with this parameter.
- If the --hyparam option is used, it takes precedence.

#### *"simulation_path"*
- Path where simulation data is stored.

#### *"potential_enabled"*
- Enable potentials

#### *"potential_grad_transition_enabled"*
- Enable state transitions with gradient along potential (only enabled if potential_enabled=true)

#### *"potential_nn_enabled"*
- Enable neural nets on potentials (only enabled if potential_enabled=true)

#### *"save_result_train"/"save_result_test"/"save_result_filter"*
- File to save the results


#### *"plot_path"*
- Path to save the plot

#### *"log"*
- Name of the file to save the log


#### *"emission_internal_layers"*
- Set the architecture of the neural network from state space to observation

#### *"transition_internal_layers"*
- Sets the architecture of a neural network from state space (time t) to state space (time t+1)


#### *"variational_internal_layers "*
- Set architecture of neural network from observation to state space

#### *"potential_internal_layers "*
- Sets the architecture of neural networks from state space to potential.

#### *"state_type "*
- Settings for the type of distribution (continuous/discrete) of the state space
- Specify either normal/discrete

#### *"sampling_type "*
- Specifies how the state space is sampled.
- If state_type=normal, specify either none/normal- If state_type=discrete, specify one of none/gambel-max/gambel-softmax/naive#### *"dynamics_type "*- Settings related to the time evolution of the state space
- Specify one of distribution/function- distribution: Construct a neural net that outputs the parameters of the distribution
- function: Construct a neural net that represents a state transition function.

#### *"pfilter_type "*
- Configuration for particle filter- Specify one of trained_dynamics/zero_dynamisc
- trained_dynamics: use trained state transitions- zero_dynamisc: use state transitions with mean zero variance 1#### *"emission_type "*- Settings for observed data
- Specify either normal/binary

#### *"pfilter_sample_size "*
- Settings related to particle filters- Sample size in particle filter state space (after resampling)

#### *pfilter_proposal_sample_size*
- Settings related to particle filters- Sample size in particle filter state space (before resampling)

#### *"pfilter_save_sample_num "*- Settings related to particle filters
- Number of samples in state space of particle filter to be saved.- Used to save only a part of the samples, since saving all of them would generate a huge amount of data when the number of samples is large.

##### *"curriculum_alpha "*
- Whether the parameter alpha is changed during training or not.
- Start with a smaller value and move closer to the final set alpha.

##### *"epoch_interval_save "*
- Specify how many epochs to save the model being trained

##### *"epoch_interval_print "*
- Specify how many epochs to print the training progress.

##### *"sampling_tau "*
Parameters for gumbel-softmax

##### *"normal_max_var "*
Maximum value of variance of the normal distribution
(Applies to all normal distributions in the model)

##### *"normal_min_var "*
Minimum value of variance of the normal distribution
(applies to all normal distributions in the model)

##### *"zero_dynamics_var "*
The magnitude of variance in a particle filter without dynamics.
(Only used when `pfilter_type="zero_dynamics"`)



# Output files
## train.jbl/test.jbl
- `z_params`: parameters of the distribution of states inferred from observations: number of data x time step x list of parameters whose elements are the number of data x time step x state space dimension
  - ["z_params"][0]=>mean: number of data x time step x state space dimension
  - ["z_params"][1]=>variance: number of data x time step x state space dimension
- `z_s`: points sampled from the distribution of states estimated from observations: number of data x time step x state space dimension
- `z_pred_params`: parameters of the next state of the state estimated from the observation: number of data x time step x state space dimension List of the number of parameters whose elements are
  - ["z_pred_params"][0]=>Average: number of data x time step x state space dimension
  - ["z_pred_params"][1]=>variance: number of data x time step x state space dimension
- `obs_params`: parameters of the distribution of observations estimated backward from the state.
  - ["obs_params"][0]=> mean: number of data x time step x observation space dimension
  - ["obs_params"][1]=> variance: number of data x time step x observation space dimension
- `obs_pred_params`: parameters of the distribution of observations estimated in the reverse direction from the state at the previous time.
  - ["obs_pred_params"][0]=> mean: number of data x time step x observation space dimension
  - ["obs_pred_params"][1]=> variance: number of data x time step x observation space dimension
- `config`.

## filter.jbl
Filtering results
- z (10, 20, 100, 2): number of particles in state space x number of data x time step x state space dimension
- mu (100, 20, 100, 1): number of particles (to be stored) x number of data x time step x observation space dimension
- error: number of particles x number of data x time step x observation space dimension

（The number of particles (to be saved) is specified by "pfilter_save_sample_num" in config.
Also, the number of particles in the state space is specified by "pfilter_sample_size".

## field.jbl
File output in field mode: Outputs the transition movement in the state space
- "z": coordinates of each grid point: number of points x state space dimension
- "gz": vector of transitions at each grid point x state space dimension


