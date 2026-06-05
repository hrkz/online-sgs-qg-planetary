import tqdm

import numpy as np
import scipy
import jax.numpy as jnp

from typing import Optional, Tuple

def into_m(f_g: jnp.ndarray, n_m: int) -> jnp.ndarray:
    """Transform grid values into Fourier coefficients."""
    f_m = jnp.fft.rfft (f_g, axis=0, norm='forward')[:n_m]
    return f_m
def from_m(f_m: jnp.ndarray, n_phi: int) -> jnp.ndarray:
    """Transform back Fourier coefficients on the grid, with 3/2 dealiasing."""
    f_f = jnp.pad(f_m, ((0, n_phi//2 + 1 - f_m.shape[0]), (0, 0)))
    f_g = jnp.fft.irfft(f_f, axis=0, norm='forward')
    return f_g
def coef_r(f_m: jnp.ndarray) -> jnp.ndarray:
    """Transform between grid values and Chebyshev coefficients (equivalent Type I DCT)."""
    n = len(f_m) - 1
    c_n = jnp.sqrt(0.5 / n)
    f_mrv = jnp.take(f_m, jnp.arange(1, n))
    f_mrv = jnp.flip(f_mrv)
    f_r = jnp.take(jnp.fft.fft(jnp.concatenate([f_m, f_mrv])), jnp.arange(n + 1))
    return c_n * f_r

def diff_r(f_m: jnp.ndarray) -> jnp.ndarray:
    """Differentiate on the Gauss-Lobatto grid (Chebyshev coefficients)."""
    n = len(f_m) - 1
    x = jnp.cos(jnp.pi * jnp.arange(0, n + 1) / n)
    c_n = 2 # 2 / (s_o - s_i)
    k_r = jnp.fft.fftfreq(2 * n, 0.5 / n).at[n].set(0)
    i_r = jnp.arange(0, n)
    
    f_mrv = jnp.take(f_m, jnp.arange(1, n))
    f_mrv = jnp.flip(f_mrv)
    f_r = jnp.fft.fft(jnp.concatenate([f_m, f_mrv]))
    df_r = jnp.fft.ifft(1j * k_r * f_r)

    df_m = c_n * jnp.zeros_like(f_m, dtype=jnp.complex128) \
        .at[1:n].set(-df_r[1:n] / jnp.sqrt(1 - x[1:n]**2)) \
        .at[0]  .set(jnp.sum(i_r**2 * f_r[:n]) / float(n) + 0.5 * n * f_r[n]) \
        .at[n]  .set(jnp.sum(((-1)**(i_r + 1))*(i_r**2) * f_r[:n]) / float(n) + 0.5 * (-1)**(n + 1) * n * f_r[n])
    return df_m
def quad_r(f: jnp.ndarray) -> float:
    """Compute the definite integral over the Gauss-Lobatto grid using quadrature."""
    n = len(f) - 1
    f_r = coef_r(0.5 * f)
    f_r = f_r.at[ 0].mul(0.5)
    f_r = f_r.at[-1].mul(0.5)
    
    w = jnp.arange(0, n + 1, 2)
    n_i = 2.0 * jnp.sqrt(2.0 / n)
    return n_i * jnp.sum(
        -1.0 / (w**2 - 1.0) * f_r[w]
    )

# Hankel kernels for Bessel-Fourier transforms
def hankel_kernels(
    eq,
    n_m_max: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    from scipy.special import jv, yn, hankel1
    """Compute the Hankel support function roots and kernels for the Bessel-Fourier transform."""
    # For m > msafety, the roots of the annulus can be approximated
    # by only retaining the type I Bessel terms
    msafety = 200
    # This is the cut-off value (x*s_i < safety*m) which allows to
    # truncate the expansion
    safety = 0.8

    if n_m_max is None:
        n_m_max = eq.n_m
    n_roots = eq.n_s // 2
    
    def jnyn_zeros(m, n_roots):
        def f(x):
            if m > msafety and x * eq.s_i < safety * m:
                return -jv(m, x * eq.s_o)
            else:
                return (jv(m, x * eq.s_o) * yn(m, x * eq.s_i) - jv(m, x * eq.s_i) * yn(m, x * eq.s_o)) / \
                        np.abs(hankel1(m, x * eq.s_i))
    
        def __full_annulus__(x):
            return (jv(m, x * eq.s_o) * yn(m, x * eq.s_i) - jv(m, x * eq.s_i) * yn(m, x * eq.s_o)) / \
                    np.abs(hankel1(m, x * eq.s_i))
        def __simplified_annulus__(x):
            return -jv(m, x * eq.s_o)

        if m <= 15:
            dx = 0.1
            xmin = 0.1
            xmax = 100
            n_evals = 1024
        else:
            dx = 0.2
            # Fit for the first root
            xmin = int(1.00 / eq.s_o * m)
            # Fit for the fifth root
            xmax = int(1.06 / eq.s_o * m + 30)
            n_evals = int((xmax - xmin) / dx)
        x = np.linspace(xmin, xmax, n_evals)
        if m > msafety:
            mask = (x * eq.s_i < safety * m)
            y = np.zeros_like(x)
            y[mask] = __simplified_annulus__(x[mask])
            y[~mask] = __full_annulus__(x[~mask])
        else:
            y = __full_annulus__(x)
        xmax = x[-1]
        signs = np.sign(y)
        roots = []
        for i in range(n_evals - 1):
            if signs[i] + signs[i + 1] == 0:
                sol = scipy.optimize.root_scalar(f, bracket=[x[i], x[i + 1]], method='brentq')
                if sol.converged and abs(sol.root) > 0.1:
                    roots.append(sol.root)
        while len(roots) < n_roots:
            dz = roots[len(roots) - 1] - roots[len(roots) - 2]
            start, stop = roots[len(roots) - 1] + 0.5*dz, roots[len(roots) - 1] + 1.5*dz
            sol = scipy.optimize.root_scalar(f, bracket=[start, stop], method='brentq')
            roots.append(sol.root)
        return np.array(
            roots
        )

    k0 = jnyn_zeros(0, n_roots + 1)
    kmax = k0[-1]
    m_roots = []
    m_kr = []
    kernels = []

    m_bar = tqdm.tqdm(range(n_m_max), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for m in m_bar:
        roots = jnyn_zeros(m, n_roots)
        kr = roots[roots <= kmax]

        x_ker = np.outer(eq.s_grid[::-1], kr)
        if m > msafety:
            mask_k = (kr * eq.s_i < safety * m)
            k_yn = kr[~mask_k]
            kernel = np.zeros_like(x_ker)

            kernel[:, mask_k] = -jv(m, x_ker[:, mask_k])
            kernel[:, ~mask_k] = (jv(m, x_ker[:, ~mask_k]) * \
                                  yn(m, eq.s_i * k_yn)  -  \
                                  yn(m, x_ker[:, ~mask_k]) * \
                                  jv(m, eq.s_i * k_yn)) / \
                                  np.abs(hankel1(m, eq.s_i * k_yn))
        else:
            kernel = (jv(m, x_ker) * yn(m, eq.s_i * kr) - yn(m, x_ker) * jv(m, eq.s_i * kr)) / \
                      np.abs(hankel1(m, eq.s_i * kr))

        m_roots.append(np.array(roots))
        m_kr.append(kr)
        kernels.append(kernel.T)
        
    return (
        m_roots,
        m_kr,
        kernels
    )
