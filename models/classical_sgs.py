import jax
import jax.numpy as jnp

from models.qg_annulus import (
    QgAnnulus
)

from utils import (
    into_m, 
    from_m, 
    diff_r,
)

def cyl_leith(
    eq: QgAnnulus,
    lam: float,
    ps_m: jnp.ndarray, 
    us_m: jnp.ndarray, 
    up_m: jnp.ndarray, 
    om_m: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    om_s = from_m(jax.vmap(diff_r)(om_m), eq.n_phi)
    om_p = from_m(1j*eq.m * om_m / eq.s_grid, eq.n_phi)

    delta_p = 2 * jnp.pi / 3 / eq.n_m
    delta_s = jnp.pad(jnp.minimum(eq.s_grid[:-2] - eq.s_grid[1:-1], eq.s_grid[1:-1] - eq.s_grid[2:]), (1,), mode='edge')
    delta = jnp.minimum(eq.s_grid * delta_p, delta_s)
    nu_e = (lam * delta / jnp.pi)**3 * jnp.sqrt(om_s**2 + om_p**2)
    
    # limiter
    nu_max = 0.25 * delta**2 / dt
    nu_e = jnp.minimum(nu_e, nu_max)

    nu_e_om_s = into_m(eq.s_grid * nu_e * om_s, eq.n_m)
    nu_e_om_p = into_m(            nu_e * om_p, eq.n_m)
    
    up_0 = up_m[0].real
    dup_0  = diff_r(up_0).real

    tau_om = 1j*eq.m * nu_e_om_p / eq.s_grid + jax.vmap(diff_r)(nu_e_om_s) / eq.s_grid
    tau_p0 = diff_r(eq.s_grid * jnp.mean(nu_e, axis=0) * dup_0).real / eq.s_grid

    return jnp.where(eq.m != 0,
                     tau_om,
                     tau_p0)
