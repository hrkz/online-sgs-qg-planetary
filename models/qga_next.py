import jax
import jax.numpy as jnp
from flax import nnx

from typing import Callable, List, Tuple

# Activations
def mod_relu(
    z: jnp.ndarray, 
    b: float = 1.0, 
    c: float = 1e-3
) -> jnp.ndarray:
    abs_z = jnp.abs(z)
    return nnx.relu(abs_z + b) * z / (abs_z + c)
def cx_gelu(z: jnp.ndarray) -> jnp.ndarray:
    return nnx.gelu(z.real) + 1j*nnx.gelu(z.imag)

class NextBlock(nnx.Module):
    def __init__(
        self, 
        activation: Callable,
        in_features: int,
        out_features: int,
        kernel_size: List[int], 
        rngs,
    ):
        self.features = out_features
        self.conv = nnx.Conv(in_features, out_features, kernel_size, rngs=rngs)
        self.lin_1 = nnx.Linear(out_features, out_features, rngs=rngs)
        self.act = activation
        self.lin_2 = nnx.Linear(out_features, out_features, rngs=rngs)
        
    def __call__(self, x):
        x = self.conv(x)
        x_id = x
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        x += x_id
        return x
    
class QgaNext(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        blocks: List[Tuple[int, int]],
        means: jnp.ndarray,
        stds: jnp.ndarray,
        activation: Callable,
        rngs
    ):
        self.means = nnx.Variable(means)
        self.stds = nnx.Variable(stds)
        self.blocks = nnx.List()
        cur_features = in_features
        for (kernel, features) in blocks:
            self.blocks.append((
                NextBlock(activation, cur_features, features, (kernel,), rngs=rngs),       # up_0 block
                NextBlock(activation, cur_features, features, (kernel, kernel), rngs=rngs) # om_m block
            ))
            cur_features = features
        self.lin_up_0 = nnx.Linear(cur_features, out_features, rngs=rngs)
        self.lin_om_m = nnx.Linear(cur_features, out_features, rngs=rngs)
    
    def __call__(self, f):
        f = (f - self.means) / self.stds
        up_0, om_m = jnp.split(f, [1], axis=1)
        for (up_0_block, om_m_block) in self.blocks:
            up_0 = up_0_block(up_0)
            om_m = om_m_block(om_m)
        up_0 = self.lin_up_0(up_0)
        om_m = self.lin_om_m(om_m)
        return jnp.concatenate(
            (up_0, om_m), 
            axis=1
        )
