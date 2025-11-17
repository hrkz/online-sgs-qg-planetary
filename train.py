import shutil
import argparse
import tqdm
import h5py
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jnr

jax.config.update(
  'jax_enable_x64', True
)

from flax import nnx
import optax
import orbax.checkpoint as ocp

import models.imex_solver as imex
from models.qg_annulus import (
    QgAnnulus, 
    dynamical_solver,
    cartesian_forcing,
)
from models.qga_next import (
    QgaNext,
    mod_relu,
    cx_gelu
)
from utils import (
    quad_r
)

def main(args: argparse.Namespace) -> None:
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.config)
    eq, *_ = QgAnnulus.load(os.path.join(data_path, 'snapshot.h5'))
    print(eq)

    with h5py.File(os.path.join(data_path, args.name + '_dataset.h5'), 'r') as f:
        dt = f.attrs['dt']
        steps = f.attrs['steps']
        coarse_factor = f.attrs['coarse_factor']

        n_m_coarse = int((eq.n_m - 1) / coarse_factor) + 1
        n_s_coarse = int((eq.n_s - 1) / coarse_factor) + 1
        eq_coarse = QgAnnulus(
            E=eq.E,
            cte_beta=eq.cte_beta,
            radius_ratio=eq.s_i / eq.s_o,
            n_m=n_m_coarse,
            n_s=n_s_coarse
        )
    
        cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)
        
        t_0 = np.array(f['t_0'][:])
        # (samples, features, m, r) -> (samples, m, r, features)
        f_m = np.moveaxis(
            np.array(f['f_m'][:]), 
             1, 
            -1)

    f_means = np.mean(f_m, axis=(0, 1, 2))
    f_stds  = np.std (f_m, axis=(0, 1, 2))
    print('''Dataset statistics: 
        ps_m = {:.4} ± σ({:.4})
        us_m = {:.4} ± σ({:.4})
        up_m = {:.4} ± σ({:.4})
        om_m = {:.4} ± σ({:.4})'''.format(
            f_means[0], f_stds[0],
            f_means[1], f_stds[1],
            f_means[2], f_stds[2],
            f_means[3], f_stds[3],
    ))
    # (samples, m, r, features) -> (sub_trajs, steps, m, r, features)
    sub_trajs = int(len(f_m) / steps)
    f_m = np.array(np.split(f_m, sub_trajs, axis=0))

    rngs = nnx.Rngs(123)
    key = rngs.params()
    eq_model = QgaNext(
        in_features=3, # (us_m, up_m, om_m)
        out_features=1, # tau_m
        blocks=[(7, 32), (7, 64), (7, 128)],
        means=jnp.array(f_means[1:]), 
        stds=jnp.array(f_stds[1:]),
        activation=mod_relu,
        rngs=rngs
    )
    #print(eq_model)

    optimizer = nnx.Optimizer(eq_model, optax.adamw(args.lr), wrt=nnx.Param)

    def flow_loss(
        eq_model: nnx.Module, 
        t_0: jnp.ndarray, 
        f_0: jnp.ndarray, 
        f_m: jnp.ndarray
    ) -> float:
        def __source__(
            _ps_m, 
            us_m: jnp.ndarray, 
            up_m: jnp.ndarray, 
            om_m: jnp.ndarray
        ) -> jnp.ndarray:
            tau_m = eq_model(jnp.expand_dims(jnp.stack((us_m, up_m, om_m), axis=-1), 0)).squeeze()
            return cf_m + tau_m
            
        solver = dynamical_solver(
            eq_coarse,
            imex.BPR353(dt * coarse_factor),
            __source__
        )
        
        def __loop__(
            states: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            ps_m, us_m, up_m, om_m = states
            c, *states = solver(ps_m, us_m, up_m, om_m)
            return (
                jnp.array(states), 
                jnp.stack(states, axis=-1)
            )
            
        _, f_m_hat = jax.lax.scan(__loop__, jnp.moveaxis(f_0, -1, 0), length=steps - 1)
        return kinetic_energy_loss(
            f_m_hat,
            f_m,
            eq_coarse.s_grid
        )

    @nnx.jit 
    def train_step(eq_model, optimizer, traj_batch: jnp.ndarray):
        t_0, f_0, f_m = traj_batch
        loss, grads = nnx.value_and_grad(flow_loss)(eq_model, t_0, f_0, f_m)
        optimizer.update(eq_model, grads)
        return loss

    train_loss = []
    print('Training model...')
    pbar = tqdm.tqdm(range(args.epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        key, subkey = jnr.split(key)
        data_sample = batch_gen(
            t_0, 
            f_m, 
            sub_trajs, 
            shuffle=True, 
            key=subkey
        )

        t_loss = 0.0
        for traj in range(sub_trajs):
            traj_batch = next(data_sample)
            loss = train_step(eq_model, optimizer, traj_batch)
            t_loss += loss / (steps - 1)

            pbar.set_postfix(
                sub_traj=traj,
                loss=t_loss / (traj + 1)
            )
        train_loss.append(t_loss / sub_trajs)
        
    np.savez(os.path.join(data_path, args.name + '_loss.npz'), loss=train_loss)
    print('Saving model parameters...')
    _, state = nnx.split(eq_model)
    checkpoint_path = os.path.join(data_path, args.name + '_checkpoint/')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        checkpoint_path,
        state
    )

# Batch generator
def batch_gen(
    t_0: np.ndarray, 
    f_m: np.ndarray, 
    sub_trajs: int,
    shuffle=True,  
    key=None
):
    """Generate trajectories (batch of continous samples) from a given dataset."""
    idx = np.arange(sub_trajs)
    if shuffle:
        idx = jnr.permutation(
            key, 
            idx,
            independent=True
        )
    
    for batch in range(sub_trajs):
        curr_idx = idx[batch]
        batch_inputs = f_m[curr_idx, 0]
        batch_target = f_m[curr_idx, 1:]
    
        yield (
            t_0[curr_idx],
            jnp.array(batch_inputs), 
            jnp.array(batch_target)
        )

# Loss
def kinetic_energy_loss(
    f_m_hat: jnp.ndarray, 
    f_m: jnp.ndarray,
    s_grid: jnp.ndarray
) -> float:
    _, us_m_hat, up_m_hat, _ = jnp.moveaxis(f_m_hat, -1, 0)
    _, us_m,     up_m,     _ = jnp.moveaxis(f_m,     -1, 0)

    us_mis_m = us_m_hat - us_m
    us2_mis = us_mis_m[:, 0].real**2 + 2*jnp.sum(us_mis_m[:, 1:].real**2 + us_mis_m[:, 1:].imag**2, axis=1)
    us2_mis_int = jnp.pi * jax.vmap(quad_r)(us2_mis * s_grid).real
    up_mis_m = up_m_hat - up_m
    up2_mis = up_mis_m[:, 0].real**2 + 2*jnp.sum(up_mis_m[:, 1:].real**2 + up_mis_m[:, 1:].imag**2, axis=1)
    up2_mis_int = jnp.pi * jax.vmap(quad_r)(up2_mis * s_grid).real
    return jnp.mean(
        us2_mis_int + up2_mis_int
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python train.py',
        description='Train a model using `online learning` from a pre-computed dataset'
    )
    
    parser.add_argument('-c', '--config', type=str, help='Name of the configuration', required=True)
    parser.add_argument('-n', '--name', type=str, help='Name of the dataset', required=True)

    parser.add_argument('-dx_f', type=float, default=0.08, help='Cartesian forcing: pump spacing')
    parser.add_argument('-radius_f', type=float, default=0.04, help='Cartesian forcing: pump radius')
    parser.add_argument('-amp_f', type=float, default=2e10, help='Cartesian forcing: amplitude')

    parser.add_argument('-lr', type=float, help='Training learning rate', required=True)
    parser.add_argument('-epochs', type=int, help='Number of training epochs', required=True)
    
    args = parser.parse_args()
    main(args)
