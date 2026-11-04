> This repository contains a JAX implementation for the paper ["Online learning of subgrid-scale models for quasi-geostrophic turbulence in planetary interiors"](https://doi.org/10.1103/lcd6-p52d) published in Physical Review Fluids (PRF). It can be used to reproduce the results presented in the manuscript.

---

## 📦 Getting started

To set up and run the Python scripts and notebooks, we use [uv](https://docs.astral.sh/uv/) to manage package dependencies in a custom environment

1. **Set up the environment**

```bash
cd online-sgs-qg-planetary
uv sync
```

Note: you need access to a GPU device because the default dependencies are based on the CUDA version of JAX. Running the code on a CPU is possible, but the `pyproject.toml` file must be modified accordingly.

## 🚀 Reproducing results

Below are the steps used to produce the results and figures from the paper. For the considered numerical resolutions, a device with at least 40 GB of (V)RAM may be required. The parameters used throughout this example correspond to *configuration (i)* of the paper.

### Setting-up the configuration

To generate a dataset for the learning pipeline, we first need to set up a configuration and run a simulation until a steady state is reached. This can be done using the following command:

```bash
uv run _snapshot.py -n i -E 2e-7 -cte_beta -1 -n_m 400 -n_s 321 -dt 5e-8 -T 0.01
```

Once the script finishes, a `snapshot.h5` file is created in the `data/i/` directory. We can now launch the notebook `notebooks/config_stats.ipynb` to analyze the configuration statistics and determine the timescales required for the dataset generation. By running the cells up to the `get_stats` function, we obtain the turnover time $t_{L}$, the number of sub-trajectories required for continuous sampling (as used in the paper), and the corresponding number of samples for dataset generation in `dataset.py`. Here, each sub-trajectory is defined to be $N_{\text{steps}} = 25$ discrete timesteps long.

The final cell can be used to generate decorrelation plots for an ensemble of perturbed simulations.

### Generating the coarse-grained dataset

We now have all the necessary ingredients to build a dataset for sub-trajectories of 25 timesteps at a resolution 5 times coarser than the reference. For this configuration, one turnover spans approximately $1.64 \times 10^{-4}$ time units, which can be divided into 27 sub-trajectories, yielding a total of 675 samples. Let's run the following command:

```bash
uv run _dataset.py -c i -n turnover -dt 5e-8 -samples 675 -steps 25 -coarse_factor 5
```

### Training the model

At this point, the `turnover` dataset saved in `data/i/` can be used to train the neural network. By default, it uses the architecture described in the paper and learns an implicit subgrid-scale correction to the dynamical system with respect to the coarse-grained fields. We launch the training for 50 epochs with a learning rate of $1 \times 10^{-4}$:

```bash
uv run _train.py -c i -n turnover -lr 1e-4 -epochs 50
```

Once training is complete, the checkpoint is saved in `data/i/` and the model parameters can be used for inference in a simulation.

### Evaluating the model(s)

Finally, we can evaluate the trained model and compare its performance against the reference DNS and other baselines. We run the evaluation script for 100 turnovers, corresponding to $100 \times 1.64 \times 10^{-4} = 0.0164$ for the *configuration (i)* and save 5000 samples:

```bash
uv run _eval.py -c i -n turnover \
    -hdiff_md 56 -hdiff_amp 1.1 -leith_lam 2.0 \
    -dt_dns 6e-8 -dt_0 7e-8 -dt_hdiff 7e-8 -dt_leith 8e-8 -dt_learn 5e-7 \
    -timespan 0.0164 -samples 5000 -save_path "my_path/"
```

**Note:** make sure to replace `my_path` with a directory that has sufficient storage space for the snapshots. Otherwise, you can reduce the number of saved samples.

We can then compute and visualize some metrics in the `notebooks/eval_metrics.ipynb` notebook. By default, the metrics are set for *configuration (i)*. To reproduce the figure from the paper, you must also run the pipeline for configurations *(ii)* and *(iii)* and modify the notebooks accordingly.

## 📖 Citing

```bibtex
@article{frezat2026online,
  title={Online learning of subgrid-scale models for quasigeostrophic turbulence in planetary interiors},
  author={Frezat, Hugo and Gastine, Thomas and Fournier, Alexandre},
  journal={Physical Review Fluids},
  volume={11},
  number={10},
  pages={104904},
  year={2026},
  publisher={APS},
  doi={10.1103/lcd6-p52d}
}

```
