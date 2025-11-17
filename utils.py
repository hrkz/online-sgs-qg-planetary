import tqdm
import matplotlib.pyplot as plt

import numpy as np
import scipy
import jax
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
    """Compute the Hankel support function roots and kernels for the Bessel-Fourier transform."""
    if n_m_max is None:
        n_m_max = eq.n_m
        
    n_roots = eq.n_s
    m_roots = np.zeros((n_m_max, eq.n_s))
    kernels = np.zeros((n_m_max, eq.n_s, eq.n_s))
    
    def __dirichlet__(x, m):
        jyc = scipy.special.jv(m, x * eq.s_o) * scipy.special.yn(m, x * eq.s_i)
        jyi = scipy.special.jv(m, x * eq.s_i) * scipy.special.yn(m, x * eq.s_o)
        return (jyc - jyi) / np.abs(scipy.special.hankel1(m, x * eq.s_i))
    def __support__(root, grid, m):
        jyc = scipy.special.jv(m, root * grid) * scipy.special.yn(m, root * eq.s_i)
        jyi = scipy.special.jv(m, root * eq.s_i) * scipy.special.yn(m, root * grid)
        return (jyc - jyi) / np.abs(scipy.special.hankel1(m, root * eq.s_i))
    
    m_bar = tqdm.tqdm(range(n_m_max), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for m in m_bar:
        if m <= int(n_m_max / 30):
            xmin, xmax = 0.1, 100
        else:
            xmin, xmax = int(1.00 / eq.s_o * m), int(1.06 / eq.s_o * m + 30)
        x = np.linspace(xmin, xmax, eq.n_s)
        y0 = __dirichlet__(x, m)
        signs = np.sign(y0)
        roots = []
        for r in range(eq.n_s - 1):
            root_interval = signs[r] + signs[r + 1] == 0
            if root_interval:
                sol = scipy.optimize.root_scalar(__dirichlet__, args=(m,), bracket=[x[r], x[r + 1]], method='brentq')
                if sol.converged and abs(sol.root) > 0.1:
                    roots.append(sol.root)
        while len(roots) < eq.n_s + 1:
            dz = roots[len(roots) - 1] - roots[len(roots) - 2]
            start, stop = roots[len(roots) - 1] + 0.5*dz, roots[len(roots) - 1] + 1.5*dz
            sol = scipy.optimize.root_scalar(__dirichlet__, args=(m,), bracket=[start, stop], method='brentq')
            roots.append(sol.root)
        m_roots[m] = np.array(roots[:-1])
        for k, root in enumerate(m_roots[m]):
            kernels[m, k] = __support__(root, eq.s_grid[::-1], m)
    return (
        m_roots,
        kernels
    )

def plot_annulus(
    eq,
    f_m: jnp.ndarray,
    cmap: str, 
    label: str, 
    vmin: float = None,
    vmax: float = None,
    levels: int = 95
):
    """Plot a field in an annulus."""
    f = from_m(f_m, eq.n_phi)
    sf = np.append(f, np.expand_dims(f[0, :], axis=0), axis=0)
    
    ph = np.linspace(0.0, 2 * np.pi, eq.n_phi + 1)
    rd = eq.s_grid
    
    pp, rr = np.meshgrid(ph, rd, indexing='ij')
    xx = rr * np.cos(pp)
    yy = rr * np.sin(pp)
    
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(4.0, 3.0), constrained_layout=True)
    if vmin is None:
        f_contour = axs.contourf(xx, yy, sf, 
                                 cmap=cmap, levels=levels)
    else:
        cs = np.linspace(vmin, vmax, levels)
        f_contour = axs.contourf(xx, yy, sf, cs, 
                                 cmap=cmap, extend='both')

    fig.colorbar(f_contour, ax=axs, shrink=0.75, aspect=40, format='%.2e')
    axs.plot(rd[ 0] * np.cos(ph), rd[ 0] * np.sin(ph), 'k-', lw=1.5)
    axs.plot(rd[-1] * np.cos(ph), rd[-1] * np.sin(ph), 'k-', lw=1.5)
    
    axs.axis('off')
    axs.set_xlim(1.01 * xx.min(), 1.01 * xx.max())
    axs.set_ylim(1.01 * yy.min(), 1.01 * yy.max())
    axs.text(0, 0, label, fontsize=15, va='center', ha='center')
    return fig
