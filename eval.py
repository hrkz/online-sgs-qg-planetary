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
import orbax.checkpoint as ocp

from typing import Callable, Optional

import models.imex_solver as imex
from models.qg_annulus import (
    QgAnnulus, 
    dynamical_solver,
    cartesian_forcing,
    galerkin_coarse_graining,
)
from models.qga_next import (
    QgaNext,
    mod_relu,
)

def main(args: argparse.Namespace) -> None:
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.config)
    eq, time, ps_m, us_m, up_m, om_m = QgAnnulus.load(os.path.join(data_path, args.name + '_snapshot.h5'))
    print(eq)

    save_path = os.path.join(args.save_path, args.config)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with h5py.File(os.path.join(data_path, args.name + '_dataset.h5'), 'r') as f:
        coarse_factor = f.attrs['coarse_factor']

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
                imex.BPR353(args.dt_dns),
                source
            ))

            _, time_s, start_ps_m, start_us_m, start_up_m, start_om_m = QgAnnulus.load(os.path.join(data_path, 'snapshot.h5'))
            print(eq)

            iters = int(np.ceil((args.timespan + (time - time_s)) / args.dt_dns))
            sample_times = np.linspace(time_s, time + args.timespan, args.samples + 1)
            run_evaluation(
                name='DNS',
                solver=solver,
                iters=iters,
                sample_times=sample_times[1:],
                t0=time_s,
                dt=args.dt_dns,
                states=(
                    start_ps_m, start_us_m, start_up_m, start_om_m
                ),
                file=f
            )   

    # LES models
    file_0 = os.path.join(
        save_path, 
        args.name + '_eval_0.h5'
    )
    if not os.path.isfile(file_0):
        with h5py.File(file_0, 'w') as f:
            eq_coarse = QgAnnulus(
                E=eq.E,
                cte_beta=eq.cte_beta,
                radius_ratio=eq.s_i / eq.s_o,
                n_m=int((eq.n_m - 1) / coarse_factor) + 1,
                n_s=eq.n_s
            )
            
            cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)
            def source(
                ps_m: jnp.ndarray, 
                us_m: jnp.ndarray, 
                up_m: jnp.ndarray, 
                om_m: jnp.ndarray
            ) -> jnp.ndarray:
                return cf_m

            ps_mc, us_mc, up_mc, om_mc = galerkin_coarse_graining(eq, eq_coarse, ps_m, up_m[0])
    
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                imex.BPR353(args.dt_0),
                source
            ))

            iters = int(np.ceil(args.timespan / args.dt_0))
            sample_times = np.linspace(time, time + args.timespan, args.samples + 1)
            run_evaluation(
                name='`Under-resolved` model',
                solver=solver,
                iters=iters,
                sample_times=sample_times[1:],
                t0=time,
                dt=args.dt_0,
                states=(
                    ps_mc, us_mc, up_mc, om_mc
                ),
                file=f
            )

    file_hdiff = os.path.join(
        save_path, 
        args.name + '_eval_hdiff.h5'
    )
    if not os.path.isfile(file_hdiff):
        with h5py.File(file_hdiff, 'w') as f:
            eq_coarse = QgAnnulus(
                E=eq.E,
                cte_beta=eq.cte_beta,
                radius_ratio=eq.s_i / eq.s_o,
                n_m=int((eq.n_m - 1) / coarse_factor) + 1,
                n_s=eq.n_s
            )
            
            cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)
            def source(
                ps_m: jnp.ndarray, 
                us_m: jnp.ndarray, 
                up_m: jnp.ndarray, 
                om_m: jnp.ndarray
            ) -> jnp.ndarray:
                return cf_m

            ps_mc, us_mc, up_mc, om_mc = galerkin_coarse_graining(eq, eq_coarse, ps_m, up_m[0])

            h_diff = jnp.where(eq_coarse.m > args.hdiff_md, args.hdiff_amp**(eq_coarse.m - args.hdiff_md), 1.0)
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                imex.BPR353(args.dt_hdiff),
                source,
                h_diff
            ))

            iters = int(np.ceil(args.timespan / args.dt_hdiff))
            sample_times = np.linspace(time, time + args.timespan, args.samples + 1)
            run_evaluation(
                name='`Hyperdiffusivity` model',
                solver=solver,
                iters=iters,
                sample_times=sample_times[1:],
                t0=time,
                dt=args.dt_hdiff,
                states=(
                    ps_mc, us_mc, up_mc, om_mc
                ),
                file=f
            )

    file_leith = os.path.join(
        save_path,
        args.name + '_eval_leith.h5'
    )
    if not os.path.isfile(file_leith):
        with h5py.File(file_leith, 'w') as f:
            eq_coarse = QgAnnulus(
                E=eq.E,
                cte_beta=eq.cte_beta,
                radius_ratio=eq.s_i / eq.s_o,
                n_m=int((eq.n_m - 1) / coarse_factor) + 1,
                n_s=eq.n_s
            )

            cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)
            from models.classical_sgs import cyl_leith
            def source(
                ps_m: jnp.ndarray,
                us_m: jnp.ndarray,
                up_m: jnp.ndarray,
                om_m: jnp.ndarray
            ) -> jnp.ndarray:
                return cf_m + cyl_leith(eq_coarse, args.leith_lam, ps_m, us_m, up_m, om_m, dt_coarse)

            ps_mc, us_mc, up_mc, om_mc = galerkin_coarse_graining(eq, eq_coarse, ps_m, up_m[0])

            solver = jax.jit(dynamical_solver(
                eq_coarse,
                imex.BPR353(args.dt_leith),
                source,
            ))

            iters = int(np.ceil(args.timespan / args.dt_leith))
            sample_times = np.linspace(time, time + args.timespan, args.samples + 1)
            run_evaluation(
                name='`Leith` model',
                solver=solver,
                iters=iters,
                sample_times=sample_times[1:],
                t0=time,
                dt=args.dt_leith,
                states=(
                    ps_mc, us_mc, up_mc, om_mc
                ),
                file=f
            )

    file_learn = os.path.join(
        save_path, 
        args.name + '_eval_learn.h5'
    )
    if not os.path.isfile(file_learn):
        with h5py.File(file_learn, 'w') as f:
            eq_coarse = QgAnnulus(
                E=eq.E,
                cte_beta=eq.cte_beta,
                radius_ratio=eq.s_i / eq.s_o,
                n_m=int((eq.n_m - 1) / coarse_factor) + 1,
                n_s=int((eq.n_s - 1) / coarse_factor) + 1
            )
            
            cf_m = cartesian_forcing(eq_coarse, args.dx_f, args.radius_f, args.amp_f)

            abstract_model = nnx.eval_shape(lambda: QgaNext(
                in_features=3, # (us_m, up_m, om_m)
                out_features=1, # tau_m
                blocks=[(7, 32), (7, 64), (7, 128)],
                means=jnp.zeros((3,), dtype=jnp.complex128), 
                stds=jnp.zeros((3,), dtype=jnp.complex128),
                activation=mod_relu,
                rngs=nnx.Rngs(42)
            ))

            graph, abstract_state = nnx.split(abstract_model)

            checkpoint_path = os.path.join(data_path, args.name + '_checkpoint/')
            checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
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

            ps_mc, us_mc, up_mc, om_mc = galerkin_coarse_graining(eq, eq_coarse, ps_m, up_m[0])
            
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                imex.BPR353(args.dt_learn),
                source,
            ))

            iters = int(np.ceil(args.timespan / args.dt_learn))
            sample_times = np.linspace(time, time + args.timespan, args.samples + 1)
            run_evaluation(
                name='`Learned` model',
                solver=solver,
                iters=iters,
                sample_times=sample_times[1:],
                t0=time,
                dt=args.dt_learn,
                states=(
                    ps_mc, us_mc, up_mc, om_mc
                ),
                file=f,
                #compute_tau=tau,
            )
            

def run_evaluation(
    name: str,
    solver: Callable,
    iters: int,
    sample_times: int,
    t0: float,
    dt: float,
    states: jnp.ndarray,
    file,
    compute_tau: Optional[Callable] = None
):
    time = t0
    ps_m, us_m, up_m, om_m = states
    
    eval_time = []

    sample_digits = len(str(len(sample_times)))
    sample_idx = 0
    print('Running evaluation for ' + name + '...')
    pbar = tqdm.tqdm(range(iters), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:            
        c, ps_m, us_m, up_m, om_m = solver(ps_m, us_m, up_m, om_m)
        if not np.isfinite(c):
            print(name + ' evaluation crashed with cfl =',c)
            return False
        time += dt

        if sample_idx < len(sample_times) and time >= sample_times[sample_idx] - 0.5*dt:
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
                sample_idx
            )
            sample_idx += 1
    file.create_dataset('time', 
                        data=np.array(eval_time))
    return True
    
            
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

    parser.add_argument('-hdiff_md', type=int, default=56, help='Hyperdiffusivity starting wavenumber')
    parser.add_argument('-hdiff_amp', type=float, default=1.1, help='Hyperdiffusivity coefficient')
    parser.add_argument('-leith_lam', type=float, default=2.0, help='Leith non-dimensional coefficient')

    parser.add_argument('-dt_dns', type=float, help='Timestep for the DNS')
    parser.add_argument('-dt_0', type=float, help='Timestep for the under-resolved simulation')
    parser.add_argument('-dt_hdiff', type=float, help='Timestep for the hyperdiffusivity simulation')
    parser.add_argument('-dt_leith', type=float, help='Timestep for the Leith model simulation')
    parser.add_argument('-dt_learn', type=float, help='Timestep for the learned model simulation')

    parser.add_argument('-timespan', type=float, help='Temporal span of the evaluation', required=True)
    parser.add_argument('-samples', type=int, help='Number of saved statistical samples', required=True)
    
    parser.add_argument('-save_path', type=str, help='Path of the directory used to save samples', required=True)
    
    args = parser.parse_args()
    main(args)
