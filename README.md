# Structural Autoencoders and Latent Responses

This code implements Structural Autoencoders and the latent response framework to learn and analyze disentangled representations for generative modeling. The paper associated with this project will be available soon.

## Setup

All models are implemented with `pytorch`, and training/evaluation results are viewable with `tensorboard`. The code is used primarily on Ubuntu and Windows and requires Python 3.7+ (it is recommended to use Anaconda).

All dependencies are listed in `requirements.txt`. The most significant dependency is `omnilearn`, which contains training loop infrastructure and many utilities to create, train, and evaluate deep models. This repository on the other hand contains the methods and analysis specifically discussed in our submitted paper, so, eventhough the code cannot be executed without `omnilearn`, it can still be read to check any precise details in our work.

To install all requirements (it is recommended to use a fresh virtual environment):

```bash
pip install -r requirements
```


## Contents

All the code is contained in the `sae_src/` directory:

- `datasets.py` - while the main datasets (3dshapes, mpi3d, celeba) we use are already implemented in `omnilearn`, this file implements a few simple datasets that don't require any downloads for demos.

- `decoders.py` - this implements a generalized structural decoder architecture used for the SAE, AdaAE, and VLAE decoders.

- `evaluate.py` - includes scripts (`eval-metrics` and `eval-multiple-metrics`) to computes the disentanglement metrics given one or multiple completed runs, respectively.

- `ladder.py` - implements the necessary components for the VLAE baseline, including the ladder encoder (`ladder-enc`), the "inference rung" used by the encoder (`rung-infer`), and the "generative rung" used by the decoder (`rung-gen`) (here the `style-dec` from `decoders.py` decoder is used for the decoder).

- `methods.py` - this contains the different regularization methods, most importantly the autoencoder (AE), variational autoencoder (VAE), and the wasserstein autoencoder (WAE)

- `responses.py` - contains the code used to compute the latent response and latent factor-response matrix (by `metric/responses` in `evaluate.py`).

- `run.py` - defines the default naming convention used for trained models based on the model regularization and architecture.

- `structure_modules.py` - contains the structural transform layer (here called `adain-affine`) and a reimplementation of the adaptive instance normalization layer.

- `baseline.py` - our reimplementation of the encoder and decoder architectures used by default in [`disentanglement_lib`](https://github.com/google-research/disentanglement_lib).

## Usage

There are three scripts used to produce all the results in the paper (the first two of which are imported from `omnilearn`):

- `train` - creates a new model and trains it

- `eval` - evaluates the reconstruction and generation performance of a trained model (on the validation or test set)

- `eval-metrics` - evaluates the disentanglement metrics and latent responses of a trained model

Since this project uses `omnifig`, the general command to run any of these scripts is the same:


```bash
fig [script-name] [config files...] --[command-line arguments...]
```

(more details can be seen by running `fig -h`)

Perhaps most importantly, you can provide any number of config files in the command which will then be combined. All config files can be found under `config/`, where the directories each focus on a different part of the script:

- `a/` - separate encoder (`e/`) and decoder (`d/`) architectures, as well as configs for combined architectures as required by VLAE `ladder/`

- `d/` - the dataset that should be used

- `m/` - the method (AE, VAE, WAE, etc.) that should be used

- `eval/` - configs for loading and computing the metrics on trained models


### Example commands

To make sure everything is setup correctly after installing the dependencies, a simple model can be trained using (note that all commands should be run from this directory):

```bash
fig train demo
```

or equivalently:

```bash
python main.py train demo
```

By default, trained model checkpoints and results are saved to a directory called `trained_nets/` in the current directory.

Provided the demo runs without issue, you can specify the config combination of your choice to try out any of the architectures/methods discussed in the paper:

```bash
# SAE-12
fig train hybrid a/e/conv12 a/d/strc/12 m/ae d/vec

# SAE-6
fig train hybrid a/e/conv12 a/d/strc/6 m/ae d/vec

# AdaAE-12
fig train hybrid nosplit a/e/conv12 a/d/strc/12 m/ae d/vec

# VLAE-12
fig train hybrid a/ladder/12 m/vae d/vec

# VLAE-6
fig train hybrid a/ladder/6 m/vae d/vec

# AE
fig train hybrid a/conv12 m/ae d/vec

# VAE
fig train hybrid a/conv12 m/vae d/vec

# beta-VAE
fig train hybrid a/conv12 m/vae d/vec --reg-wt 2

# WAE
fig train hybrid a/conv12 m/wae d/vec
```

All the above commands use a toy dataset generated on the fly using a randomly initialized network called `d/vec`, but to run the more interesting image datasets discussed in the paper, they must be downloaded and formatted as expected by `omnilearn`. 

To download and format the datasets automatically you can use the `--download` flag (or the `download-dataset` script from `omnilearn`). When downloading, all datasets are by default saved to a local directory called `local_data/`. The follow command will run a simple autoencoder and download the corresponding datasets:

```bash
# 3D-Shapes
fig train hybrid a/conv12 m/ae d/3ds --download

# MPI-3D Toy
fig train hybrid a/conv12 m/ae d/mpi --download --cat toy

# MPI-3D Realistic (aka Sim)
fig train hybrid a/conv12 m/ae d/mpi --download --cat realistic

# MPI-3D Real
fig train hybrid a/conv12 m/ae d/mpi --download --cat real

# Celeb-A (note that here the 16 layer architectures must be used for Celeb-A)
fig train hybrid a/conv16 m/ae d/celeba --download
```

Note that 3dshapes is about 250 MB, while each of the MPI3D datasets are >10 GB, and CelebA is about 2GB).

Once the model is trained, it can be loaded for evaluation using the `--load` argument, to evaluate on the testset the `--mode` can be set, and using `--ident` you can specify the name of the saved evaluation results. For example, running the demo saves a run in `trained_nets/` named something like `vec_ae_e-c12_d-c12_210210-142525` which can then be loaded/evaluated:


```bash
# Will save a `val_results.pth.tar` file in the run directory
fig eval --load vec_ae_e-c12_d-c12_210210-142525 --ident val_results

# Will evaluate results on test set
fig eval --load vec_ae_e-c12_d-c12_210210-142525 --mode test --ident test_results
```

Lastly, the disentanglement metrics can be evaluated for the 3dshapes and MPI3d dataset using the `eval-metrics` script:

```bash
fig eval-metrics eval/3ds --load 3ds_ae_e-c12_d-c12_210210-153556
```


