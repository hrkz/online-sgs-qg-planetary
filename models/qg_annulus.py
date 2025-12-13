import h5py
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jax.scipy as jns

from typing import Callable, Optional, Tuple

from models.imex_solver import Scheme
from utils import (
    into_m, 
    from_m, 
    coef_r, 
    diff_r,
    quad_r
)

class QgAnnulus:
    def __init__(
        self, 
        E: float, 
        cte_beta: float,
        radius_ratio: float,
        n_m: int,
        n_s: int
    ):
        self.E = E
        self.cte_beta = cte_beta
        self.s_i = radius_ratio / (1 - radius_ratio)
        self.s_o = 1 / (1 - radius_ratio)
        self.n_m = n_m
        self.n_phi = 3 * self.n_m
        self.n_s = n_s

        self.m = np.expand_dims(np.arange(0, self.n_m), axis=1)
        self.gl_grid = np.cos(np.pi * np.arange(0, self.n_s) / (self.n_s - 1))
        self.s_grid = 0.5 * (self.s_o - self.s_i) * self.gl_grid + 0.5 * (self.s_o + self.s_i)

        if self.cte_beta != 0:
            # Exponential container (constant beta)
            self.beta     = self.cte_beta / self.s_o
            self.dbeta_ds = 0
            self.height   = np.exp(self.beta * self.s_grid)
            self.epump    = (1 + self.beta**2 * self.height**2)**0.25 / (self.E**0.5 * self.height)
        else:
            # Spherical container (varying beta)
            hsq = self.s_o**2 - self.s_grid**2
            self.height = np.sqrt(hsq)
            with np.errstate(divide='ignore'):
                # Handle singularity at the outer boundary
                self.beta     = np.where(self.height != 0, -self.s_grid / hsq, 0)
                self.dbeta_ds = np.where(self.height != 0, -(self.s_o**2 + self.s_grid**2) / hsq**2, 0)
                self.epump    = np.where(self.height != 0, self.s_o**0.5 / (self.E**0.5 * hsq**0.75), 0)

        self.surf = np.pi * (self.s_o**2 - self.s_i**2)
        # Delta p/s(s)
        self.dx_dp2 = (2 * np.pi / 3 / self.n_m * self.s_grid)**2
        self.dx_ds2 = np.pad(np.minimum(self.s_grid[:-2] - self.s_grid[1:-1], self.s_grid[1:-1] - self.s_grid[2:])**2, (1,), mode='edge')
        
        (self.T_d2r, 
         self.T_dr, 
         self.T_r) = self.__coll_recurrence__()

    def __coll_recurrence__(self):
        """Create (dense) collocation matrices."""
        gl_dr = 2 / (self.s_o - self.s_i)
        gl_norm = np.sqrt(2 / (self.n_s - 1))
        
        T_r = np.zeros((self.n_s, self.n_s), dtype=np.float64)
        T_dr = np.zeros((self.n_s, self.n_s), dtype=np.float64)
        T_d2r = np.zeros((self.n_s, self.n_s), dtype=np.float64)        
        # Collocation matrices (recurrences)
        T_r[:, 0] = 1
        T_r[:, 1] = self.gl_grid
        T_dr[:, 0] = 0
        T_dr[:, 1] = gl_dr
        T_d2r[:, 0] = 0
        T_d2r[:, 1] = 0
        for n in range(2, self.n_s):
            T_r[:, n] = 2 * self.gl_grid * T_r[:, n - 1] - T_r[:, n - 2]
            T_dr[:, n] = 2 * gl_dr * T_r[:, n - 1] + 2 * self.gl_grid * T_dr[:, n - 1] - T_dr[:, n - 2]
            T_d2r[:, n] = 4 * gl_dr * T_dr[:, n - 1] + 2 * self.gl_grid * T_d2r[:, n - 1] - T_d2r[:, n - 2]
        return (
            gl_norm * T_d2r,
            gl_norm * T_dr,
            gl_norm * T_r
        )

    def save(self, filename: str, time: float, ps_m, us_m, up_m, om_m):
        with h5py.File(filename, 'w') as f:
            f.attrs['E'] = self.E
            f.attrs['cte_beta'] = self.cte_beta
            f.attrs['radius_ratio'] = self.s_i / self.s_o
            f.attrs['n_m'] = self.n_m
            f.attrs['n_s'] = self.n_s
            
            f.attrs['time'] = time
            
            f.create_dataset('ps_m',
                             data=np.array(ps_m))
            f.create_dataset('us_m',
                             data=np.array(us_m))
            f.create_dataset('up_m',
                             data=np.array(up_m))
            f.create_dataset('om_m',
                             data=np.array(om_m))

    def load(filename: str):
        with h5py.File(filename, 'r') as f:
            eq = QgAnnulus(
                E=f.attrs['E'].item(), 
                cte_beta=f.attrs['cte_beta'].item(), 
                radius_ratio=f.attrs['radius_ratio'].item(),
                n_m=f.attrs['n_m'].item(),
                n_s=f.attrs['n_s'].item()
            )
            
            return (
                eq,
                f.attrs['time'].item(),
                np.array(f['ps_m']),
                np.array(f['us_m']),
                np.array(f['up_m']),
                np.array(f['om_m'])
            )

    def __repr__(self):
        return '''Quasi-geostrophic system: 
        Parameters: E={}, cte_beta={} on radius [{:.4},{:.4}]
        Truncation: n_m={} (n_phi={}), n_s={}'''.format(
            self.E,
            self.cte_beta,
            self.s_i,
            self.s_o,
            self.n_m,
            self.n_phi,
            self.n_s
        )

def eq_system(
    eq: QgAnnulus,
    coef: float, 
    h_diff: jnp.ndarray
):
    # Promote to column vector
    def _c(y):
        return jnp.expand_dims(
            y, axis=1
        )

    up0_mat = eq.T_d2r + _c(1 / eq.s_grid) * eq.T_dr - _c(1 / eq.s_grid**2 + eq.epump) * eq.T_r
    up0_sys = eq.T_r - coef * h_diff[0] * up0_mat

    up0_sys = up0_sys.at[ 0].set(eq.T_r[ 0]) # dirichlet
    up0_sys = up0_sys.at[-1].set(eq.T_r[-1]) # dirichlet
    up0_sys = up0_sys.at[:,  0].mul(0.5)
    up0_sys = up0_sys.at[:, -1].mul(0.5)
    
    def __coupled_psi_sys__(m):
        om_om_mat = eq.T_d2r +  _c(1 / eq.s_grid) * eq.T_dr - _c(m**2 / eq.s_grid**2 + eq.epump) * eq.T_r
        om_om_block = eq.T_r - coef * h_diff[m] * om_om_mat
        om_ps_block = jnp.zeros_like(om_om_block)
        ps_om_block = jnp.array(eq.T_r)
        ps_ps_block = eq.T_d2r + _c(1 / eq.s_grid + eq.beta) * eq.T_dr + _c(eq.beta / eq.s_grid + eq.dbeta_ds - m**2 / eq.s_grid**2) * eq.T_r
          
        om_om_block = om_om_block.at[ 0].set(0.0)
        om_om_block = om_om_block.at[-1].set(0.0)
        om_ps_block = om_ps_block.at[ 0].set(eq.T_r[ 0]) # dirichlet
        om_ps_block = om_ps_block.at[-1].set(eq.T_r[-1]) # dirichlet
        ps_om_block = ps_om_block.at[ 0].set(0.0)
        ps_om_block = ps_om_block.at[-1].set(0.0)
        ps_ps_block = ps_ps_block.at[ 0].set(eq.T_dr[ 0]) # neumann
        ps_ps_block = ps_ps_block.at[-1].set(eq.T_dr[-1]) # neumann
        
        om_ps_sys = jnp.block([
          [om_om_block, 
           om_ps_block], 
          [ps_om_block, 
           ps_ps_block]
        ])
        
        # factors
        om_ps_sys = om_ps_sys.at[:,          0].mul(0.5)
        om_ps_sys = om_ps_sys.at[:, eq.n_s - 1].mul(0.5)
        om_ps_sys = om_ps_sys.at[:,     eq.n_s].mul(0.5)
        om_ps_sys = om_ps_sys.at[:,         -1].mul(0.5)
        return jns.linalg.lu_factor(
            om_ps_sys
        )
      
    return (
        jns.linalg.lu_factor(up0_sys), 
        jax.vmap(__coupled_psi_sys__)(eq.m)
    )
    
def dynamical_solver(
    eq: QgAnnulus,
    solver: Scheme,
    source: Callable,
    h_diff: Optional[jnp.ndarray] = None
) -> Callable:
    if h_diff is None: 
        h_diff = jnp.ones_like(eq.m)
        
    # Courant–Friedrichs–Lewy condition (number).
    def __cfl__(
        us_g: jnp.ndarray,
        up_g: jnp.ndarray
    ) -> float:
        us2_max = jnp.max(us_g**2, axis=0)
        up2_max = jnp.max(up_g**2, axis=0)
        
        dt_s = jnp.min(jnp.sqrt(eq.dx_ds2 / us2_max))
        dt_p = jnp.min(jnp.sqrt(eq.dx_dp2 / up2_max))
        return jnp.min(
            0.75 * jnp.array([jnp.finfo(jnp.float64).max, dt_s, dt_p])
        )

    # Implicit term
    def __implicit__(
        us_m: jnp.ndarray,
        up_m: jnp.ndarray,
        om_m: jnp.ndarray
    ) -> jnp.ndarray:
        up_0 = up_m[0].real
        dup_0  = diff_r(up_0).real
        dom_m  = jax.vmap(diff_r)(om_m)
        d2up_0 = diff_r(dup_0).real
        d2om_m = jax.vmap(diff_r)(dom_m)

        return jnp.where(eq.m != 0,
                         h_diff * (d2om_m + dom_m / eq.s_grid - (eq.m**2 / eq.s_grid**2 + eq.epump) * om_m),
                         h_diff * (d2up_0 + dup_0 / eq.s_grid - (      1 / eq.s_grid**2 + eq.epump) * up_0)
                        )

    # Explicit term
    def __explicit__(
        us_m: jnp.ndarray,
        up_m: jnp.ndarray,
        om_m: jnp.ndarray,
        source: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        us_g = from_m(us_m, eq.n_phi)
        up_g = from_m(up_m, eq.n_phi)
        om_g = from_m(om_m, eq.n_phi)
        
        c = __cfl__(us_g, up_g)
        
        usom_g = eq.s_grid * us_g * om_g
        upom_g =             up_g * om_g
        
        upom_m = into_m(upom_g, eq.n_m)
        usom_m = into_m(usom_g, eq.n_m)
        
        dusom_m = jax.vmap(diff_r)(usom_m)

        self_i = (eq.E / 2) * eq.epump * up_m[0] * om_m[0]
        iusom_0 = self_i.real + 2*jnp.sum(us_m[1:].real * om_m[1:].real + us_m[1:].imag * om_m[1:].imag, axis=0)
        
        return jnp.where(eq.m != 0,
                         -1j*eq.m * upom_m / eq.s_grid - dusom_m / eq.s_grid + (2 / eq.E) * eq.beta * us_m + source,
                         -iusom_0 + source
                         ), c

    # Implicit solve
    def __solve__(
        eq_system: Tuple[jnp.ndarray, jnp.ndarray],
        rhs: jnp.ndarray
    ) -> jnp.ndarray:
        up0_mat_lu, om_ps_mats_lu = eq_system
        
        up0_eq = rhs[0]
        up0_eq = up0_eq.at[ 0].set(0.0) # up0 (s=si) = 0
        up0_eq = up0_eq.at[-1].set(0.0) # up0 (s=so) = 0
        up_0_r = jns.linalg.lu_solve(up0_mat_lu, up0_eq.real)
        
        om_ps_eq = jnp.zeros((eq.n_m, 2 * eq.n_s), dtype=np.complex128).at[:, :eq.n_s].set(rhs)
        om_ps_eq = om_ps_eq.at[:,          0].set(0.0) # psi (s=si) = 0
        om_ps_eq = om_ps_eq.at[:, eq.n_s - 1].set(0.0) # psi (s=so) = 0
        om_ps_eq = om_ps_eq.at[:,     eq.n_s].set(0.0) # dpsi/dr (s=si) = 0
        om_ps_eq = om_ps_eq.at[:,         -1].set(0.0) # dpsi/dr (s=so) = 0
        om_ps_r_real = jax.vmap(jns.linalg.lu_solve)(om_ps_mats_lu, om_ps_eq.real)
        om_ps_r_imag = jax.vmap(jns.linalg.lu_solve)(om_ps_mats_lu, om_ps_eq.imag)
        om_ps_r = om_ps_r_real + 1j*om_ps_r_imag
        
        om_r, ps_r = jnp.split(om_ps_r, 2, axis=1)
        
        up_0 = coef_r(up_0_r)
        om_0 = diff_r(up_0) + up_0 / eq.s_grid
        
        ps_m = jax.vmap(coef_r)(ps_r)
        dps_m = jax.vmap(diff_r)(ps_m)
        
        om_m = jax.vmap(coef_r)(om_r)
        om_m = om_m.at[0].set(om_0)
        us_m = jnp.where(eq.m != 0,
                         1j*eq.m * ps_m / eq.s_grid,
                         0.0
                         )
        up_m = jnp.where(eq.m != 0,
                         -dps_m - eq.beta * ps_m,
                         up_0
                         )
        return (
            ps_m,
            us_m,
            up_m,
            om_m
        )
    system = eq_system(
        eq,
        solver.coef(),
        h_diff
    )
    return solver(
        eq,
        source,
        system,
        __implicit__,
        __explicit__,
        __solve__,
    )

def init_u(eq: QgAnnulus, amp_u: float) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize to a non-zero azimuthal velocity with a small perturbation."""
    up_m = np.zeros((eq.n_m, eq.n_s), dtype=np.complex128)
    up_m[0] = amp_u * np.sin(np.pi * (eq.s_grid - eq.s_i))
    dup_m = jax.vmap(diff_r)(up_m)
    om_m = dup_m + up_m / eq.s_grid
    return (
        up_m, 
        om_m
    )

def cartesian_forcing(
    eq: QgAnnulus,
    dx_f: float, 
    radius_f: float, 
    amp_f: float
) -> np.ndarray:
    """
    Cartesian forcing described in
    
    Zonal jets experiments in the gas giants’ zonostrophic regime.
    D. Lemasquerier, B. Favier and M. Le Bars.
    Icarus 390 (2023).
    """
    f_m = np.zeros((eq.n_m, eq.n_s), dtype=np.complex128)
    nx = int(2 * eq.s_o / dx_f + 1)
    ny = nx
    dx = 2 * eq.s_o / (nx - 1)
    dy = dx
    
    x_lins, y_lins = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    amp_grid = (-1)**(x_lins + 1) * (-1)**(y_lins + 1)
    x_grid, y_grid = np.meshgrid(-eq.s_o + dx * np.arange(nx), -eq.s_o + dy * np.arange(ny), indexing='ij')
    iso_grid = np.sqrt(x_grid*x_grid + y_grid*y_grid)
    
    pump_position = (iso_grid >= eq.s_i + 0.5 * dx) & \
                    (iso_grid <= eq.s_o - 0.5 * dx)
    
    amp = amp_grid[pump_position]
    x = x_grid[pump_position]
    y = y_grid[pump_position]
    
    x_phi_grid = eq.s_grid * np.expand_dims(np.cos(2 * np.pi * np.arange(eq.n_phi) / eq.n_phi), axis=1)
    y_phi_grid = eq.s_grid * np.expand_dims(np.sin(2 * np.pi * np.arange(eq.n_phi) / eq.n_phi), axis=1)
    
    n_pumps = len(amp)
    if n_pumps % 2 != 0: n_pumps -= 1
    f_g = np.zeros((eq.n_phi, eq.n_s), dtype=np.float64)
    for pump_i in range(n_pumps):
        f_g += amp_f * amp[pump_i] * np.exp(-(x[pump_i] - x_phi_grid)**2 / radius_f**2) * np.exp(-(y[pump_i] - y_phi_grid)**2 / radius_f**2)
    return into_m(f_g, eq.n_m).at[0].set(0)

def galerkin_coarse_graining(
    eq: QgAnnulus,
    eq_coarse: QgAnnulus,
    ps_m: jnp.ndarray, 
    up_0: jnp.ndarray
):
    """Coarse-grain the QG states using Galerkin bases."""
    nbc_d = 2 # dirichlet
    def __dirichlet__(n_g: int):
        T_r2 = np.diag(jnp.ones(n_g - 2), k=-2)
        T_r = np.eye(n_g)
        G_d = T_r2 - T_r
        G_d[ 0] *= 2
        G_d[-1] *= 2
        return G_d
    nbc_dn = 4 # dirichlet + neumann
    def __dirichlet_neumann__(n_g: int):
        T_r = np.eye(n_g)
        T_r2 = np.diag(2 * np.arange(2, n_g) / (np.arange(2, n_g) + 1), k=-2)
        T_r4 = np.diag((np.arange(4, n_g) - 3) / (np.arange(4, n_g) - 1), k=-4)
        G_dn = T_r - T_r2 + T_r4
        G_dn[ 0] *= 2
        G_dn[-1] *= 2
        return G_dn

    norm_r_coarse = 1 / np.sqrt((eq.n_s - 1) / (eq_coarse.n_s - 1))

    up_0r = coef_r(up_0)
    up_0g = jnp.linalg.solve(__dirichlet__(eq.n_s)[nbc_d:, :eq.n_s - nbc_d], up_0r[nbc_d:])
    up_0c = up_0g[:eq_coarse.n_s - nbc_d]
    up0_rc = norm_r_coarse * jnp.dot(__dirichlet__(eq_coarse.n_s), np.pad(up_0c, (0,nbc_d)))
    up0_mc = coef_r(up0_rc)

    ps_mt = ps_m[:eq_coarse.n_m]
    ps_rt = jax.vmap(coef_r)(ps_mt)
    ps_gt = jax.vmap(jnp.linalg.solve, (None, 0))(__dirichlet_neumann__(eq.n_s)[nbc_dn:, :eq.n_s - nbc_dn], ps_rt[:, nbc_dn:])
    ps_gc = ps_gt[:, :eq_coarse.n_s - nbc_dn]
    ps_rc = norm_r_coarse * jax.vmap(jnp.dot, (None, 0))(__dirichlet_neumann__(eq_coarse.n_s), jnp.pad(ps_gc, ((0,0), (0,nbc_dn))))
    ps_mc = jax.vmap(coef_r)(ps_rc)

    dps_mc = jax.vmap(diff_r)(ps_mc)

    om_mc = -jax.vmap(diff_r)(dps_mc) - \
            (eq_coarse.beta + 1 / eq_coarse.s_grid) * dps_mc - \
            (eq_coarse.beta / eq_coarse.s_grid + eq_coarse.dbeta_ds - eq_coarse.m**2 / eq_coarse.s_grid**2) * ps_mc
    om_mc = om_mc.at[0].set(diff_r(up0_mc) + up0_mc / eq_coarse.s_grid)

    us_mc = jnp.where(eq_coarse.m != 0,
                      1j*eq_coarse.m * ps_mc / eq_coarse.s_grid,
                      0.0
                      )
    up_mc = jnp.where(eq_coarse.m != 0,
                      -dps_mc - eq_coarse.beta * ps_mc,
                      up0_mc
                      )
    
    return (
        ps_mc,
        us_mc, 
        up_mc, 
        om_mc
    )

# Diagnostics

def integral(
    eq: QgAnnulus,
    f_m: jnp.ndarray
) -> float:
    """Compute the integral of f_m over the domain."""
    return 2 * np.pi * np.real(quad_r(
        eq.s_grid * (f_m[0].real**2 + 2*jnp.sum(f_m[1:].real**2 + f_m[1:].imag**2, axis=0))
    ))

def average(
    eq: QgAnnulus,
    f_m: jnp.ndarray
) -> float:
    """Compute the averaging of f_m over the domain."""
    return integral(eq, f_m) / eq.surf

def reynolds(
    eq: QgAnnulus,
    us_m: jnp.ndarray,
    up_m: jnp.ndarray
) -> float:
    """Compute the Reynolds number (Re)."""
    avg_ke = 0.5 * (average(eq, us_m) + average(eq, up_m))
    return np.sqrt(2 * avg_ke)

def azimuthal_spectrum(
    eq: QgAnnulus,
    f_m: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the azimuthal spectrum f(m)."""
    return np.real(jax.vmap(quad_r)(
        np.pi * eq.s_grid * np.where(eq.m == 0, f_m.real**2, 2 * (f_m.real**2 + f_m.imag**2))
    ))

def hankel_spectrum(
    eq: QgAnnulus, 
    m_roots: jnp.ndarray, 
    kernels: jnp.ndarray, 
    f_m: jnp.ndarray, 
    n_max_m: Optional[int] = None
) -> jnp.ndarray:
    """Compute the Bessel-Fourier spectrum f(k)."""
    if n_max_m is None:
        n_max_m = eq.n_m
    n_max_m = min(n_max_m, m_roots.shape[0])
    f_k = np.zeros((n_max_m, eq.n_s))
    for m in range(n_max_m):
        f_hat = jax.vmap(quad_r)(kernels[m] * f_m[m, ::-1] * eq.s_grid[::-1])
        vmin = scipy.special.jv(m, m_roots[m] * eq.s_o)**2
        vmax = scipy.special.jv(m, m_roots[m] * eq.s_i)**2
        f_nrm = m_roots[m]**2 * np.abs(scipy.special.hankel1(m, m_roots[m] * eq.s_o))**2
        f_nrm = np.where(vmin > np.finfo(np.float64).eps,
                         f_nrm * vmax / (vmax - vmin), 
                         f_nrm
                        )
        f_k[m] = 0.5 * np.pi**2 * f_nrm * np.abs(f_hat)**2
    f_k[0 ] *= np.pi
    f_k[1:] *= 2 * np.pi
    return f_k
