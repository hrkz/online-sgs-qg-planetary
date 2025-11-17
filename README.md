<p align="center">
<img src="https://github.com/hrkz/online-sgs-qg-planetary/blob/main/assets/graphical-abstract.png" alt="Graphical Abstract" width="400"/>
</p>

> This repository contains a JAX implementation for the paper ["Online learning of subgrid-scale models for quasi-geostrophic turbulence in planetary interiors"](#). It can be used to reproduce results presented in the manuscript.

## Getting started

To setup and run the Python scripts and notebooks, we use [uv](https://docs.astral.sh/uv/) to manage the package dependencies in a custom environment

1. **Create and activate the environment**

```bash
cd online-sgs-qg-planetary
uv init
```

2. **Install the required packages**

```bash
uv add -r requirements.txt
```

Note: you need the access to a GPU device since the default requirement packages are based on the CUDA version of JAX. Running the code on CPU is posible, but modification of the `requirements.txt` file is necessary.

# Reproducing results

Below are the steps used to produce the results and figures from the paper. For the considered numerical resolutions, a device with at least 40GB of (V)RAM may be required. The parameters used throughout this example correspond to *configuration (i)* of the paper.

## Setting-up the configuration

In order to generate a dataset for the learning stack, we first need to setup a configuration and run a simulation until a steady-state is reached. This can be done using the following command:

```bash
uv run snapshot.py -n i -E 2e-7 -cte_beta -1 -n_m 640 -n_s 321 -dt 4e-8 -T 0.01
```

Once the script finishes, a `snapshot.h5` file is created under the folder `data/i/`. We can now launch the notebook `docs/config_stats.ipynb` to analyse the statistics of the configuration and thus determine the timescales for the dataset generation. Running the script until the `get_stats` function, we are provided with the turnover time $t_{L}$ and the number of sub-trajectories of $N_{\mathrm{steps}} = 25$ discrete timesteps per turnovers for a continuous sampling (as used in the paper). The last cell can be used to generate the decorrelation plots of an ensemble of perturbed simulations.

## Generating the coarse-grained dataset

We now have the ingredients to build a dataset for sub-trajectories of 25 timesteps and a resolution 5 times coarser than the reference. For this configuration, a turnover spans approximately $1.7 \times 10^{-4}$ and we can thus fit 34 sub-trajectories. Lets run the following command:

```bash
uv run dataset.py -c i -n continuous-turnover -dt 4e-8 -timespan 1.7e-4 -sub_trajs 34 -steps 25 -coarse_factor 5
```

## Training the model

At this point, the `continuous-turnover` dataset saved in `data/i/` can be used to train our neural network. By default, it uses the architecture described in the paper and learn an implicit subgrid-scale correction to the dynamical system with respect to the coarse-grained fields. We launch the training for 200 epochs with a learning rate of $2 \times 10^{-5}$:

```bash
uv run train.py -c i -n continuous-turnover -lr 2e-5 -epochs 200
```

Once finished, the training checkpoint is saved in `data/i/` and we can use the model parameters for inference in a simulation.

## Evaluating the model(s)

Finally, we want to evaluate the trained model and its performance against the reference DNS and some other baselines. We run the evaluation script for 100 turnovers, which corresponds to $100 \times 1.7 \times 10^{-4} = 0.017$ for the *configuration (i)* and save 5000 samples:

```bash
uv run eval.py -c i -n continuous-turnover -timespan 0.017 -samples 5000 -save_path 'my_path/'
```

Note: make sure to replace `my_path` with a folder on which you have enough space available to save the snapshots. Otherwise, you can reduce the number of saved samples.

We can now compute and visualise some metrics in the `docs/eval_metrics.ipynb` notebook. By default, each metric (except for the integrated quantities) is set for *configuration (i)*, but to reproduce the figure from the paper, you are required to start the pipeline for configurations *(ii)* and *(iii)* and modify the notebooks accordingly.
