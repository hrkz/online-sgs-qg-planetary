<p align="center">
  <img src="https://github.com/hrkz/online-sgs-qg-planetary/blob/main/assets/repo-abstract.png" alt="Repository Abstract" width="400"/>
</p>

> This repository contains a JAX implementation for the paper ["Online learning of subgrid-scale models for quasi-geostrophic turbulence in planetary interiors"](https://arxiv.org/abs/2511.14581) submitted to the Journal of Fluid Mechanics (JFM). It can be used to reproduce results presented in the manuscript.

---

## 📦 Getting started

To setup and run the Python scripts and notebooks, we use [uv](https://docs.astral.sh/uv/) to manage the package dependencies in a custom environment

1. **Setup the environment**

```bash
cd online-sgs-qg-planetary
uv sync
```

Note: you need the access to a GPU device since the default requirement packages are based on the CUDA version of JAX. Running the code on CPU is posible, but modification of the `pyproject.toml` file is necessary.

## 🚀 Reproducing results

Below are the steps used to produce the results and figures from the paper. For the considered numerical resolutions, a device with at least 40GB of (V)RAM may be required. The parameters used throughout this example correspond to *configuration (i)* of the paper.

### Setting-up the configuration

In order to generate a dataset for the learning stack, we first need to setup a configuration and run a simulation until a steady-state is reached. This can be done using the following command:

```bash
uv run snapshot.py -n i -E 2e-7 -cte_beta -1 -n_m 400 -n_s 321 -dt 5e-8 -T 0.01
```

Once the script finishes, a `snapshot.h5` file is created under the folder `data/i/`. We can now launch the notebook `docs/config_stats.ipynb` to analyse the statistics of the configuration and thus determine the timescales for the dataset generation. Running the cells until the `get_stats` function, we are provided with the turnover time $t_{L}$, the number of sub-trajectories of $N_{\text{steps}} = 25$ discrete timesteps per turnovers for a continuous sampling (as used in the paper) and the corresponding number of samples used for the dataset generation in `dataset.py`. The last cell can be used to generate the decorrelation plots of an ensemble of perturbed simulations.

### Generating the coarse-grained dataset

We now have the ingredients to build a dataset for sub-trajectories of 25 timesteps and a resolution 5 times coarser than the reference. For this configuration, a turnover spans approximately $1.64 \times 10^{-4}$ that can fit in 27 sub-trajectories, which gives a total of 675 samples. Lets run the following command:

```bash
uv run dataset.py -c i -n turnover -dt 5e-8 -samples 675 -steps 25 -coarse_factor 5
```

### Training the model

At this point, the `turnover` dataset saved in `data/i/` can be used to train our neural network. By default, it uses the architecture described in the paper and learn an implicit subgrid-scale correction to the dynamical system with respect to the coarse-grained fields. We launch the training for 50 epochs with a learning rate of $1 \times 10^{-4}$:

```bash
uv run train.py -c i -n turnover -lr 1e-4 -epochs 50
```

Once finished, the training checkpoint is saved in `data/i/` and we can use the model parameters for inference in a simulation.

### Evaluating the model(s)

Finally, we want to evaluate the trained model and its performance against the reference DNS and some other baselines. We run the evaluation script for 100 turnovers, which corresponds to $100 \times 1.64 \times 10^{-4} = 0.0164$ for the *configuration (i)* and save 5000 samples:

```bash
uv run eval.py -c i -n turnover \
    -hdiff_md 56 -hdiff_amp 1.1 -leith_lam 2.0 \
    -dt_dns 6e-8 -dt_0 7e-8 -dt_hdiff 7e-8 -dt_leith 8e-8 -dt_learn 5e-7 \
    -timespan 0.0164 -samples 5000 -save_path 'my_path/'
```

Note: make sure to replace `my_path` with a folder on which you have enough space available to save the snapshots. Otherwise, you can reduce the number of saved samples.

We can now compute and visualise some metrics in the `docs/eval_metrics.ipynb` notebook. By default, each metric is set for *configuration (i)*, but to reproduce the figure from the paper, you are required to start the pipeline for configurations *(ii)* and *(iii)* and modify the notebooks accordingly.

## 📖 Citing

Still a preprint.
