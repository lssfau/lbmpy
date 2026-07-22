from lbmpy.enums import Stencil
from lbmpy import velocityspace

import sympy as sp


class LBStencil:
    r"""
    Factory class for lattice Boltzmann stencil implementations defined in the lbmpy.velocityspace module.

    Stencils are represented in the DxQy notation, where d is the dimension (length of the velocity tuples) and y is
    the number of discrete velocities. For every dimension many different versions of a certain stencil is available.
    The reason for that is to ensure comparability with the literature. Internally the stencil is represented as a
    tuple of tuples, where the ordering of the tuples plays no role.

    Args:
        stencil:
            Can be tuple of tuples which represents the stencil, a string like 'D2Q9'/'d2q9' or an enum of
            lbmpy.enums.Stencil
        theta0:
            The lattice reference temperature. This argument must be specified for custom stencils and is not necessary
            for pre-defined stencils.
        weights:
            The lattice weights. This argument must be specified for custom stencils and is not necessary for
            pre-defined stencils.
        ordering:
            The LBM literature does not use a common order of the discrete velocities. Therefore, different common
            orderings are made available for pre-defined stencils to compare intermediate results with the literature.
            All orderings lead to the same method, they just have to be used consistently. If not provided, the default
            "walberla" ordering is used.
        temperature:
            Stencil weights and velocities are, in the most general form, functions of temperature. Stencils can,
            therefore, be provided with a temperature to evaluate these quantities during run time. If not provided,
            temperature defaults to the lattice reference temperature.
        stencil_name:
            The name of the stencil. This argument must be specified for custom stencils and is not necessary for
            pre-defined stencils.
        stencil_type:
            lbmpy.velocityspace.StencilType enum representing the stencil type. This argument must be specified for
            custom stencils and is not necessary for pre-defined stencils.
    """

    def __new__(
        cls,
        stencil: Stencil | str | tuple[tuple[int, ...], ...],
        *,
        theta0: sp.Expr | None = None,
        weights: tuple[sp.Expr, ...] | None = None,
        ordering: str | None = None,
        temperature: sp.Expr | None = None,
        stencil_name: str | None = None,
        stencil_type: velocityspace.StencilType | None = None,
    ):
        if ordering is None:
            ordering = "walberla"

        if isinstance(stencil, velocityspace.LBStencilBase):
            # the supplied stencil is a valid pre-defined stencil object
            return stencil
        elif isinstance(stencil, str | Stencil):
            # predefined stencils are instantiated with fixed standard weights
            # for thermal weights the corresponding classes must be directly instantiated
            try:
                key = Stencil[stencil.upper()] if isinstance(stencil, str) else stencil
            except KeyError:
                raise ValueError(
                    f"Stencil '{stencil}' is not defined in lbmpy.enums.Stencil."
                ) from None

            try:
                cls = getattr(velocityspace, key.name)
            except AttributeError:
                raise NotImplementedError(
                    f"'{stencil}' is currently not implemented"
                ) from None

            return cls(ordering=ordering)
        elif isinstance(stencil, tuple):
            missing_args = [k for k, v in locals().items() if v is None]
            missing_args.remove("temperature")

            if missing_args:
                raise ValueError(
                    "Missing stencil parameter(s): {}".format(", ".join(missing_args))
                )

            return velocityspace.CustomStencil(
                velocities=stencil,
                theta0=theta0,
                weights=weights,
                ordering=ordering,
                temperature=temperature,
                stencil_name=stencil_name,
                stencil_type=stencil_type,
            )
        else:
            raise ValueError(
                "The LBStencil can only be created with either a tuple of tuples which defines the "
                "stencil, a string or an Enum of type lbmpy.enums.Stencil"
            ) from None
