# Code Accompanying: *Computational network models for forecasting and control of mental health trajectories in digital applications*
 
This repository contains the code used for the analyses in the manuscript:
 
> *Computational network models for forecasting and control of mental health trajectories in digital applications*  
> Janik Fechtelpeter, Christian Rauschenberg, Christian Goetzl, Selina Hiller, Eva Wierzba, Niklas Emonds, Silvia Krumm, Ulrich Reininghaus, Daniel Durstewitz, Georgia Koppe 
> Submitted to *npj Digital Medicine*.
 
---
 
## Status of This Code

**!!! The repo history was rewritten, please reclone the repo (`git clone fresh`) !!!**
 
This codebase is currently being prepared for public, reproducible release.  

At this stage, the **main** branch is **not** intended to be run “out of the box” by external users. Check out the **refactoring** channel for the current status of adapting the code.

The following changes have already been committed:

* Conversion into a **Python package** installable via pip.
* Replacing internal paths with **portable configuration options**.
* Refactoring into a user-friendly format (e.g., installation scripts, container images, or step-by-step workflows).
* Adding **documentation** (e.g., environment specification, dependency list, and usage instructions).

Currently, the following changes are still underway:

- Preparing **example data or synthetic data**, where appropriate, to demonstrate the main workflows.
- Integrating **trained models** to exactly reproduce the paper's findings.
- Expanding documentation and code commentary.
 
---
 
## Intended Use (Current Version)
 
At this revision stage, the **main** repository branch is primarily intended to:
 
- Document the **exact codebase** used for the analyses in the manuscript.
- Enable **editorial and peer review** of the implementation details.
 
It is **not yet** intended as a ready-to-run software package for general use or for full reproduction of all results by external users.

The **refactoring** branch already contains code that can be used to train models out-of-the-box with user-provideed data.

---

## Installation Guide

1. Git clone the **refactoring** branch onto your local machine, or download the code as a zip archive and unpack it.
2. Make sure Python (Version 3.9 or greater) is installed.
3. Open a terminal (*PowerShell* on windows) and navigate to the folder containing the package.
4. Enter `python3 -m pip install -e ema_forecast_control`. Note that the `-e` flag makes it possible to make changes to the code after it's already been installed.

---

## Usage Guide

### Quick Model Batch Training

1. Define a new project: In the *projects* folder, create a `[project name].yml` file that specifies all parameters for your model training. See below (section **Project Specification**) or the ``template.yml`` file for details on how to do that.
2. Run `python3 train_project.py [project name]` to start training. See below (section **Model Batch Training**) for further details.
3. The trained models appear in the *trained_models* folder, in the subfolder with the same name as your project. 

### Quick Model Analysis

1. All analyses are contained in Jupyter notebooks in the *analyses* folder.
2. All analyses start by specifying the `project_name` argument, which is used to load the correct models with the correct settings, and the `save` argument.
3. If `save==True`, analysis results will be saved into the *results* folder.

### Data Requirements

Data files must be stored in sub-folders of "data". Data must be in .csv format with the following format conventions:
* Each row = one timestep, chronologically ordered
* First row = column headers
* Order of columns is irrelevant

The files must contain the following columns:
* an absolute or relative time column. If absolute, it must contain date and time information, e.g. in YYYY-MM-DD hh-mm-ss format. If relative, it must contain a seconds counter relative to an anchor time.
* a *participant* column containing the participant id (must be the same in all rows)
* one column per EMA item
* one column per input item

#### Example csv file

This is how an example data file could look like. Note that EMA and input columns can have any name.

| Datetime          | Participant           | anxiety     | stress     | social_interaction   | alcohol_consumption   |
|--|--|--|--|--|--|
| 2025-01-01 09:30:00 | 42 | 4 | 5 | 0 | 1 |
| 2025-01-01 11:30:00 | 42 | 3 | 5 | 1 | 1 |
| 2025-01-01 13:30:00 | 42 | 2 | 6 | 0 | 0 |
| 2025-01-01 15:30:00 | 42 | 1 | 6 | 0 | 1 |

### Project Specification

Project files contain collections of settings for model training and evaluation. Mandatory arguments must be specified by every project file, while optional arguments are model-specific or have default values.

#### Mandatory Arguments

These arguments must be defined in every project file:

| Argument                                 | Explanation                                                                                                     |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `data_directory`                         | Directory containing the data files. Must be a sub-directory of "data".                                         |
| `obs_features`                           | List of observation features (EMA items) to be modeled                                                          |
| `input_features`                         | List of input features (control/intervention variables). Empty list or `null` if not applicable.                |
| `timestamp`                              | Timestamp configuration dictionary with the following entries:                                                  |
| `└─ absolute_datetime_column`            | Column name containing absolute datetime values (e.g., 'DateTime') (optional if `relative_datetime_column` is given)                                             |                                            |
| `└─ relative_datetime_column`            | Column name containing relative datetime values (e.g., 'Timerels') (optional if `absolute_datetime_column` is given)                                             |
| `└─ time_anchor`                         | Reference timestamp for relative times (e.g., '2022-01-01 00:00:00')                                            |
| `└─ datetime_format`                     | Format string for parsing datetime (e.g., '%Y-%m-%d %H:%M:%S').                                                  |
| `preprocessing`                          | List of preprocessing operations to apply (e.g., time_smoothing with parameters). Declare each operation as a sub-entry. In turn, declare function arguments as sub-entries to this operation.                                |
| `train_test_split`                       | Split point for training/test data. Can be:<br>• Integer: timepoint index<br>• String: datetime value<br>• List of the above: each value will be applied to each model configuration<br>• String: filename containing split information by participant |
| `model`                                  | Model type to use: 'plrnn', 'transformer', 'kalman filter', 'var1', 'linear regression', 'mean predictor', or 'last step'.                              |

#### Optional Arguments

These arguments are model-specific or have default values that can be overridden:

| Argument                      | PLRNN | Transformer | Kalman Filter | Simple Models | Explanation                                                                         |
|-------------------------------|-------|-------------|---------------|---------------|-------------------------------------------------------------------------------------|
| `train_on_last_n_steps`       | ✓     | ✓           | ✓             | ✓             | Use only the last n steps before the train-test split for training, useful to simulate shorter time series   |
| `seq_len`                     | ✓     | ✓           | ✗             | ✗             | Sequence length for training                                                        |
| `partial_missings_are_valid`  | ✓     | ✓           | ✓             | ✓             | Count partially missing EMA values as valid time steps. Not recommended.            |
| `tolerate_reduced_seq_len`    | ✓     | ✓           | ✗             | ✗             | If `seq_len` is too long for a dataset, allow reducing it automatically.            |
| `data_dropout_to_level`       | ✓     | ✓           | ✓             | ✓             | Randomly drop data points in time series until specified ratio of valid points remains. |
| `batch_size`                  | ✓     | ✓           | ✗             | ✗             | Batch size for training                         |
| `batches_per_epoch`           | ✓     | ✓           | ✗             | ✗             | Number of batches per epoch (0 = use all data)                                      |
| `n_epochs`                    | ✓     | ✓           | ✗             | ✗             | Number of training epochs                                                           |
| `learning_rate`               | ✓     | ✓           | ✗             | ✗             | Learning rate for optimizer                                                         |
| `lr_annealing`                | ✓     | ✓           | ✗             | ✗             | Whether to use learning rate annealing                                              |
| `gradient_clipping`           | ✓     | ✓           | ✗             | ✗             | Gradient clipping threshold                                                         |
| `model_save_step`             | ✓     | ✓           | ✗             | ✗             | Interval in epochs for saving model ('best', 'last' or epoch number)      |
| `info_save_step`              | ✓     | ✓           | ✗             | ✗             | Interval in epochs for saving training information                                            |
| `tf_alpha`                    | ✓     | ✗           | ✗             | ✗             | Teacher forcing alpha parameter for BPTT                                            |
| `validation_len`              | ✓     | ✓           | ✓             | ✓             | Length of validation sequences                                                      |
| `validation_prewarming`       | ✓     | ✓           | ✗             | ✗             | Prewarming steps for validation                                 |
| `early_stopping`              | ✓     | ✓           | ✗             | ✗             | Whether to use early stopping                                                       |
| `dim_z`                       | ✓     | ✗           | ✓             | ✗             | Latent dimension size                                |
| `dim_x_proj`                  | ✓     | ✗           | ✗             | ✗             | Dimension of observation projection (0 = no observation model)                      |
| `dim_y`                       | ✓     | ✗           | ✗             | ✗             | Dimension for shallow PLRNN                                                         |
| `clip_range`                  | ✓     | ✗           | ✗             | ✗             | Clipping range for PLRNN activations                                                |
| `mean_centering`              | ✓     | ✗           | ✓             | ✓             | Whether to center data by mean                                                      |
| `dim_model`                   | ✗     | ✓           | ✗             | ✗             | Transformer model dimension (must be even)                                          |
| `dim_feedforward`             | ✗     | ✓           | ✗             | ✗             | Dimension of feedforward network in transformer                                     |
| `decoder_seq_len`             | ✗     | ✓           | ✗             | ✗             | Sequence length for transformer decoder                                             |
| `n_encoder_layers`            | ✗     | ✓           | ✗             | ✗             | Number of encoder layers in transformer                                             |
| `n_decoder_layers`            | ✗     | ✓           | ✗             | ✗             | Number of decoder layers in transformer                                             |
| `n_heads`                     | ✗     | ✓           | ✗             | ✗             | Number of attention heads in transformer                                            |
| `dropout`                     | ✗     | ✓           | ✗             | ✗             | Dropout rate for transformer                                                        |
| `max_seq_len`                 | ✗     | ✓           | ✗             | ✗             | Maximum sequence length for positional encoding                                     |
| `intercept`                   | ✗     | ✗           | ✓             | ✓             | Whether to include intercept in model                                               |
| `max_A_eigval`                | ✗     | ✗           | ✓             | ✓             | Maximum eigenvalue for transition matrix A                                          |
| `impute_missing_values`       | ✗     | ✗           | ✓             | ✓             | Whether to impute missing values                                                    |

#### Hyperparameters

For hyperparameter grid search, or if you want different combinations of arguments for a different reason, it is not necessary to define a new project for each combination. You can instead define these arguments as *hyperparameters* with several values. Models will be trained for each possible combination of all hyperparameter values.

For example, say you want to grid search `learning_rate` and `dim_z`. Defining
```yaml
hyperparameters
- learning_rate:
    - 0.001
    - 0.0005
- dim_z:
    - 10
    - 20
```
will result in 4 combinations of model configurations: ``learning_rate=0.001, dim_z=10``; ``learning_rate=0.001, dim_z=20``; and so forth. All other arguments will stay the same. In YAML terms, *hyperparameters* is a list of lists.

If you would like to use some, but not all possible combinations of arguments, or you want to tune a group of hyperparameters in conjunction, you can define them as *combined_hyperparameters*. There, you can explicitly specify which combinations of arguments to use.

For example, say you want `learning_rate=0.001` and `dim_z=10` as well as `learning_rate=0.0005` and `dim_z=20`, but not the other combinations. Defining
```yaml
combined_hyperparameters
- learning_rate: 0.001
  dim_z: 10
- learning_rate: 0.0005
  dim_z: 20
``` 
will result in exactly these 2 model configurations. Again, all other arguments will stay the same.Note that each hyperparameter combinations is defined by a single `-`. In YAML terms, *combined_hyperparameters* is a list of dictionaries.


### Model batch training

Usually, you will want to train a batch of models on a dataset consisting of several participants. Hence, let's assume that your data directory `data/example` contains $n$ csv files, one per participant. Let's also assume that your project yml file is called `example_project.yml` and in it, the data_directory is specified as `example`.

#### Usage of `train_project.py`

The script `train_project.py` contained in the root directory trains one or more models on all or a subset of the participants in your data directory. The basic usage is
```
python train_project.py example_project
```
This will train one model per participant and hyperparameter configuration (hyperparameter configurations are derived from your project yml file).

You can specify the batch settings via the following optional arguments:
* `--n_runs`: number of models trained per participant. Default: 1
* `--include_participants`: list of participants to include in training; if 'none', includes all participants. Default: 'none'
* `--n_processes`: total number of parallel processes to spawn. Default: 1
* `--use_gpu`: if 0, train on CPU. If nonzero, train on GPUs if possible. Default: 0
* `--n_proc_per_gpu`: maximum number of processes spawned on one GPU. If this number times the number of available GPUs is lower than n_processes, n_processes is reduced automatically. Default: 1
* `--verbose`: logging behavior, options: `none`, `print`, and `log`. If `none`, training generates no logs. If `print`, outputs are passed to stdout (usually the console). If `log`, they are saved in a log file in the logs directory. Default: 'print' 
* `--wait`: seconds to wait until training starts. This serves the purpose of allowing for an emergency stop by pressing ctrl+C. Default: 10

#### Training device

Training can be perfomed on the CPU or Nvidia GPUs, if the appropriate CUDA version is installed on the system. Tasks are automatically assigned to the GPU with the lowest current load. The fallback is always CPU. 

#### In case the model directory already exists

Models will be saved in the directory `trained_models/<project_name>`, where `project_name` is the name of the specified project yml file. For each model, a subdirectory is created. The name of the subdirectory is formed by the hyperparameter specification of that model. 

If this directory already exists, you will be propted "Model path `<project_name>` already exists. Delete/Overwrite/Update/Abort". Here's what the options mean in detail:
* Delete: delete the whole `<project_name>` directory and create a new one.
* Overwrite: write into the existing directory, overwriting any existing subdirectories if a new model with the same hyperparameter configuration is trained.
* Update: write into the existing directory, not touching any existing subdirectories. The respective hyperparameter configurations are ignored during training.
* Abort: don't train anything.

### Model analysis

Update pending.

---

## Contact
 
If you have questions about the code or its planned public release, please contact:
 
- *Janik Fechtelpeter* – *Hector Institute for Artificial Intelligence in Psychiatry, Central Institute of Mental Health (CIMH), Medical Faculty Mannheim, Heidelberg University (UHD), Germany.*  
  Email: janik.fechtelpeter@zi-mannheim.de
 
Please mention the manuscript title (*Computational network models for forecasting and control of mental health trajectories in digital applications*) in your message.