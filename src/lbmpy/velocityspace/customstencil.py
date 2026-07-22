from lbmpy.velocityspace import StencilType, LBStencilBase
import sympy as sp


class CustomStencil(LBStencilBase):
    def __init__(
        self,
        velocities: tuple[tuple[int, ...], ...],
        theta0: sp.Expr,
        weights: tuple[sp.Expr, ...],
        stencil_name: str,
        stencil_type: StencilType,
        *,
        ordering: str | None = None,
        temperature: sp.Expr | None = None,
    ):
        self._check_valid_stencil(velocities)
        self._check_valid_weights(weights)
        self._ordering = ordering
        self.orderings = {f"{ordering}": tuple(range(len(velocities)))}
        self._theta0 = theta0

        super().__init__(
            velocities=velocities,
            theta=self.theta0 if temperature is None else temperature,
            weights=weights,
            ordering=ordering,
            stencil_name=stencil_name,
            stencil_type=stencil_type,
        )
