import argparse
import warnings
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
import orbax.checkpoint as ocp

from typing import Callable, Optional

import models.imex_solver as imex
from models.qg_annulus import (
    QgAnnulus, 
    dynamical_solver,
    cartesian_forcing,
)
from models.qga_next import (
    QgaNext,
    mod_relu,
)

def main(args: argparse.Namespace) -> None:
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.config)
    eq, time, ps_m, us_m, up_m, om_m = QgAnnulus.load(os.path.join(data_path, 'snapshot.h5'))
    print(eq)
    
    save_path = os.path.join(args.save_path, args.config)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with h5py.File(os.path.join(data_path, args.name + '_dataset.h5'), 'r') as f:
        dt = f.attrs['dt']
        steps = f.attrs['steps']
        coarse_factor = f.attrs['coarse_factor']

        dt_coarse = dt * coarse_factor
        time_coarse = f['t_0'][-1] + (steps - 1) * dt_coarse
        states_coarse = f['f_m'][-1]

        n_m_coarse = int((eq.n_m - 1) / coarse_factor) + 1
        n_s_coarse = int((eq.n_s - 1) / coarse_factor) + 1
        eq_coarse = QgAnnulus(
            E=eq.E,
            cte_beta=eq.cte_beta,
            radius_ratio=eq.s_i / eq.s_o,
            n_m=n_m_coarse,
            n_s=n_s_coarse
        )

    # DNS
    file_dns = os.path.join(save_path, args.name + '_eval_dns.h5')
    if not os.path.isfile(file_dns):
        with h5py.File(file_dns, 'w') as f:
            cf_m = cartesian_forcing(eq, args.dx_f, args.radius_f, args.amp_f) 
            def source(
                ps_m: jnp.ndarray, 
                us_m: jnp.ndarray, 
                up_m: jnp.ndarray, 
                om_m: jnp.ndarray
            ) -> jnp.ndarray:
                return cf_m
    
            solver = jax.jit(dynamical_solver(
                eq,
                imex.BPR353(dt),
                source
            ))

            iters = int((args.timespan + (time_coarse - time)) / dt)
            run_evaluation(
                name='DNS',
                solver=solver,
                iters=iters,
                sample_freq=int(iters / args.samples),
                t0=time,
                dt=dt,
                states=(
                    ps_m, us_m, up_m, om_m
                ),
                file=f
            )

    # LES models    
    file_0 = os.path.join(save_path, args.name + '_eval_0.h5')
    if not os.path.isfile(file_0):
        with h5py.File(file_0, 'w') as f:
            cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)
            def source(
                ps_m: jnp.ndarray, 
                us_m: jnp.ndarray, 
                up_m: jnp.ndarray, 
                om_m: jnp.ndarray
            ) -> jnp.ndarray:
                return cf_m
    
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                imex.BPR353(dt_coarse),
                source
            ))

            iters = int(args.timespan / dt_coarse)
            run_evaluation(
                name='`Under-resolved` model',
                solver=solver,
                iters=iters,
                sample_freq=int(iters / args.samples),
                t0=time_coarse,
                dt=dt_coarse,
                states=states_coarse,
                file=f
            )

    file_hdiff = os.path.join(save_path, args.name + '_eval_hdiff.h5')
    if not os.path.isfile(file_hdiff):
        with h5py.File(file_hdiff, 'w') as f:
            cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)
            def source(
                ps_m: jnp.ndarray, 
                us_m: jnp.ndarray, 
                up_m: jnp.ndarray, 
                om_m: jnp.ndarray
            ) -> jnp.ndarray:
                return cf_m

            h_diff = jnp.where(eq_coarse.m > args.hdiff_md, args.hdiff_amp**(eq_coarse.m - args.hdiff_md), 1.0)
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                imex.BPR353(dt_coarse),
                source,
                h_diff
            ))

            iters = int(args.timespan / dt_coarse)
            run_evaluation(
                name='`Hyperdiffusivity` model',
                solver=solver,
                iters=iters,
                sample_freq=int(iters / args.samples),
                t0=time_coarse,
                dt=dt_coarse,
                states=states_coarse,
                file=f
            )

    file_learn = os.path.join(save_path, args.name + '_eval_learn.h5')
    if not os.path.isfile(file_learn):
        with h5py.File(file_learn, 'w') as f:
            cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)

            abstract_model = nnx.eval_shape(lambda: QgaNext(
                in_features=3, # (us_m, up_m, om_m)
                out_features=1, # tau_m
                blocks=[(7, 32), (7, 64), (7, 128)],
                means=jnp.zeros((3,)), 
                stds=jnp.zeros((3,)),
                activation=mod_relu,
                rngs=nnx.Rngs(42)
            ))

            graph, abstract_state = nnx.split(abstract_model)

            checkpoint_path = os.path.join(data_path, args.name + '_checkpoint/')
            checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
            with warnings.catch_warnings(action='ignore'):
                state = checkpointer.restore(checkpoint_path, abstract_state)
            eq_model = nnx.merge(graph, state)

            def tau(
                ps_m: jnp.ndarray, us_m: jnp.ndarray, up_m: jnp.ndarray, om_m: jnp.ndarray
            ) -> jnp.ndarray:
                return eq_model(jnp.expand_dims(jnp.stack((us_m, up_m, om_m), axis=-1), 0)).squeeze()
            def source(
                ps_m: jnp.ndarray, 
                us_m: jnp.ndarray, 
                up_m: jnp.ndarray, 
                om_m: jnp.ndarray
            ) -> jnp.ndarray:
                tau_m = tau(ps_m, us_m, up_m, om_m)
                return cf_m + tau_m

            solver = jax.jit(dynamical_solver(
                eq_coarse,
                imex.BPR353(dt_coarse),
                source,
            ))

            iters = int(args.timespan / dt_coarse)
            run_evaluation(
                name='`Learned` model',
                solver=solver,
                iters=iters,
                sample_freq=int(iters / args.samples),
                t0=time_coarse,
                dt=dt_coarse,
                states=states_coarse,
                file=f,
                compute_tau=tau,
            )
            

def run_evaluation(
    name: str,
    solver: Callable,
    iters: int,
    sample_freq: int,
    t0: float,
    dt: float,
    states: jnp.ndarray,
    file,
    compute_tau: Optional[Callable] = None
):
    time = t0
    ps_m, us_m, up_m, om_m = states
    
    eval_time = []

    sample_digits = len(str(int(iters / sample_freq)))
    print('Running evaluation for ' + name + '...')
    pbar = tqdm.tqdm(range(iters), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:            
        c, ps_m, us_m, up_m, om_m = solver(ps_m, us_m, up_m, om_m)
        if not np.isfinite(c):
            print(name + ' evaluation crashed with cfl =',c)
            return
        time += dt

        if i % sample_freq == 0:
            eval_time.append(time)
            tau_m = compute_tau(ps_m, us_m, up_m, om_m) if compute_tau else None
            write_sample(
                file, 
                ps_m, 
                us_m, 
                up_m, 
                om_m, 
                tau_m,
                sample_digits, 
                i // sample_freq
            )
    file.create_dataset('time', 
                        data=np.array(eval_time))
    
            
def write_sample(
    file, 
    ps_m: jnp.ndarray, 
    us_m: jnp.ndarray, 
    up_m: jnp.ndarray, 
    om_m: jnp.ndarray, 
    tau_m: Optional[jnp.ndarray],
    sample_digits: int,
    i: int
):
    file.create_dataset('ps_m_' + str(i).zfill(sample_digits),
                        data=np.array(ps_m))
    file.create_dataset('us_m_' + str(i).zfill(sample_digits),
                        data=np.array(us_m))
    file.create_dataset('up_m_' + str(i).zfill(sample_digits),
                        data=np.array(up_m))
    file.create_dataset('om_m_' + str(i).zfill(sample_digits),
                        data=np.array(om_m))
    if tau_m != None:
        file.create_dataset('tau_m_' + str(i).zfill(sample_digits),
                            data=np.array(tau_m))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python eval.py',
        description='Evaluate the trained model against reference DNS and baselines'
    )
    
    parser.add_argument('-c', '--config', type=str, help='Name of the configuration', required=True)
    parser.add_argument('-n', '--name', type=str, help='Name of the dataset', required=True)

    parser.add_argument('-dx_f', type=float, default=0.08, help='Cartesian forcing: pump spacing')
    parser.add_argument('-radius_f', type=float, default=0.04, help='Cartesian forcing: pump radius')
    parser.add_argument('-amp_f', type=float, default=2e10, help='Cartesian forcing: amplitude')

    parser.add_argument('-hdiff_md', type=int, default=96, help='Hyperdiffusivity starting wavenumber')
    parser.add_argument('-hdiff_amp', type=float, default=1.12, help='Hyperdiffusivity coefficient')

    parser.add_argument('-timespan', type=float, help='Temporal span of the evaluation', required=True)
    parser.add_argument('-samples', type=int, help='Number of saved statistical samples', required=True)
    parser.add_argument('-save_path', type=str, help='File path for saving the field samples', required=True)
    
    args = parser.parse_args()
    main(args)
