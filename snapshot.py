import argparse
import tqdm
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
    cartesian_forcing,
)

def main(args: argparse.Namespace) -> None:
    eq = QgAnnulus(
        E=args.E,
        cte_beta=args.cte_beta,
        radius_ratio=args.rr,
        n_m=args.n_m,
        n_s=args.n_s
    )
    print(eq)

    ps_m, us_m, up_m, om_m = np.zeros((4, eq.n_m, eq.n_s), dtype=np.complex128)
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

    iters = int(args.T / dt)
    print('Running dynamical solver...')
    pbar = tqdm.tqdm(range(iters), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        c, ps_m, us_m, up_m, om_m = solver(ps_m, us_m, up_m, om_m)
        if not np.isfinite(c):
            print('Solver crashed with cfl =',c)
            exit(1)

    print('Saving snapshot...')
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    eq.save(
        os.path.join(data_path, 'snapshot.h5'),
        args.T,
        ps_m, 
        us_m, 
        up_m, 
        om_m
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python snapshot.py',
        description='Integrate a QG system configuration to time T and save a snapshot for dataset generation.'
    )
    
    parser.add_argument('-n', '--name', type=str, help='Name of the configuration', required=True)
    parser.add_argument('-E', type=float, help='Ekman number', required=True)
    parser.add_argument('-cte_beta', type=float, default=0.0, help='Exponential geometry constant beta (0 for spherical)', required=True)
    parser.add_argument('-rr', type=float, default=0.35, help='Ratio between inner and outer cylindrical radius')

    parser.add_argument('-n_m', type=int, help='Number of Fourier coefficients (azimuthal direction)', required=True)
    parser.add_argument('-n_s', type=int, help='Number of Chebyshev coefficients (radial direction)', required=True) 

    parser.add_argument('-dt', type=float, help='Discrete (fixed) time step', required=True)
    parser.add_argument('-T', type=float, help='Final time of the integration', required=True)
    
    parser.add_argument('-dx_f', type=float, default=0.08, help='Cartesian forcing: pump spacing')
    parser.add_argument('-radius_f', type=float, default=0.04, help='Cartesian forcing: pump radius')
    parser.add_argument('-amp_f', type=float, default=2e10, help='Cartesian forcing: amplitude')
    
    args = parser.parse_args()
    main(args)
