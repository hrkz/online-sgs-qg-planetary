import argparse
import tqdm
import h5py
import os

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update(
  'jax_enable_x64', True
)

import models.imex_solver as imex
from models.qg_annulus import (
    QgAnnulus, 
    dynamical_solver,
    galerkin_coarse_graining,
    cartesian_forcing,
)
from utils import (
    coef_r,
    diff_r,
)

def main(args: argparse.Namespace) -> None:
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.config)
    eq, time, ps_m, us_m, up_m, om_m = QgAnnulus.load(os.path.join(data_path, 'snapshot.h5'))
    print(eq)

    eq_coarse = QgAnnulus(
        E=eq.E,
        cte_beta=eq.cte_beta,
        radius_ratio=eq.s_i / eq.s_o,
        n_m=int((eq.n_m - 1) / args.coarse_factor) + 1,
        n_s=int((eq.n_s - 1) / args.coarse_factor) + 1
    )

    cf_m = cartesian_forcing(eq, args.dx_f, args.radius_f, args.amp_f)
    def source(
        ps_m: jnp.ndarray, 
        us_m: jnp.ndarray, 
        up_m: jnp.ndarray, 
        om_m: jnp.ndarray
    ) -> jnp.ndarray:
        return cf_m
    
    dt = args.dt
    solver = jax.jit(dynamical_solver(
        eq,
        imex.BPR353(dt),
        source
    ))

    data_samples = []
    data_t0 = []

    total_iters = int(np.round(args.timespan / args.dt))
    sub_trajs_freq = int(total_iters / args.sub_trajs)
    print('Generating dataset...')
    pbar = tqdm.tqdm(range(total_iters), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        c, ps_m, us_m, up_m, om_m = solver(ps_m, us_m, up_m, om_m)
        if not np.isfinite(c):
            print('Solver crashed with cfl =',c)
            exit(1)
        time += args.dt

        cur_sub_traj = i // sub_trajs_freq
        if cur_sub_traj == args.sub_trajs:
            break

        sub_traj_iter = i % sub_trajs_freq
        if sub_traj_iter == 0: 
            # Starting a sub-trajectory
            data_t0.append(time)
        if sub_traj_iter < args.coarse_factor * args.steps:
            if sub_traj_iter % args.coarse_factor == 0: 
                # Sample each coarse-factor iterations
                data_samples.append(galerkin_coarse_graining(
                    eq,
                    eq_coarse,
                    ps_m,
                    up_m[0]
                ))

    print('Saving dataset...')
    with h5py.File(os.path.join(data_path, args.name + '_dataset.h5'), 'w') as f:
        f.attrs['dt'] = args.dt
        f.attrs['steps'] = args.steps
        f.attrs['coarse_factor'] = args.coarse_factor

        f.create_dataset('t_0',
                         data=np.array(data_t0))
        f.create_dataset('f_m',
                         data=np.array(data_samples))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python dataset.py',
        description='Generate a dataset for `online learning` from a steady-state snapshot'
    )
    
    parser.add_argument('-c', '--config', type=str, help='Name of the configuration', required=True)
    parser.add_argument('-n', '--name', type=str, help='Name of the dataset', required=True)

    parser.add_argument('-dt', type=float, help='Discrete (fixed) time step (for DNS)', required=True)

    parser.add_argument('-dx_f', type=float, default=0.08, help='Cartesian forcing: pump spacing')
    parser.add_argument('-radius_f', type=float, default=0.04, help='Cartesian forcing: pump radius')
    parser.add_argument('-amp_f', type=float, default=2e10, help='Cartesian forcing: amplitude')
    
    parser.add_argument('-timespan', type=float, help='Temporal span of the dataset', required=True)
    parser.add_argument('-sub_trajs', type=int, help='Number of sub-trajectories', required=True)
    parser.add_argument('-steps', type=int, help='Number of discrete time steps for a sub-trajectory', required=True)

    parser.add_argument('-coarse_factor', type=int, help='Coarsening factor for grid resolution and time step', required=True)
    
    args = parser.parse_args()
    main(args)
