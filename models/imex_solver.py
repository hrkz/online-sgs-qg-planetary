import numpy as np
import jax
import jax.numpy as jnp

from typing import Callable, List, Tuple

class Scheme:
    def coef(self) -> float:
        pass
    def __call__(
        self,
        eq,
        system: jnp.ndarray,
        source: Callable,
        implicit: Callable,
        explicit: Callable,
        solve: Callable,
    ) -> Callable:
        pass

class BPR353(Scheme):
    def __init__(self, dt: float):
        self.dt = dt
    def coef(self) -> float:
        return 0.5 * self.dt
    def __call__(
        self,
        eq,
        source: Callable,
        eq_system: List[jnp.ndarray],
        implicit: Callable,
        explicit: Callable,
        solve: Callable,
    ):
        def __imex_step__(
            ps_m: jnp.ndarray,
            us_m: jnp.ndarray,
            up_m: jnp.ndarray,
            om_m: jnp.ndarray
        ) -> Tuple[float, jnp.ndarray]:
            eq_dt = jnp.where(eq.m != 0, om_m, up_m)
            
            expl_0, c = explicit(us_m, up_m, om_m, source(ps_m, us_m, up_m, om_m))
            impl_0    = implicit(us_m, up_m, om_m)
            step_0    = eq_dt + self.dt * expl_0 + self.dt * 0.5 * impl_0
            ps_m, us_m, up_m, om_m = solve(eq_system, step_0)
            
            expl_1, _ = explicit(us_m, up_m, om_m, source(ps_m, us_m, up_m, om_m))
            impl_1    = implicit(us_m, up_m, om_m)
            step_1    = eq_dt + self.dt * (4.0/9.0 * expl_0 + 2.0/9.0 * expl_1) + self.dt * (5.0/18.0 * impl_0 - 1.0/9.0 * impl_1)
            ps_m, us_m, up_m, om_m = solve(eq_system, step_1)
            
            expl_2, _ = explicit(us_m, up_m, om_m, source(ps_m, us_m, up_m, om_m))
            impl_2    = implicit(us_m, up_m, om_m)
            step_2    = eq_dt + self.dt * (0.25 * expl_0 + 0.75 * expl_2) + self.dt * 0.5 * impl_0
            ps_m, us_m, up_m, om_m = solve(eq_system, step_2)
            
            impl_3    = implicit(us_m, up_m, om_m)
            step_3    = eq_dt + self.dt * (0.25 * expl_0 + 0.75 * expl_2) + self.dt * (0.25 * impl_0 + 0.75 * impl_2 - 0.5 * impl_3)
            ps_m, us_m, up_m, om_m = solve(eq_system, step_3)
            
            return (
              c, ps_m, us_m, up_m, om_m
            )
        return __imex_step__
