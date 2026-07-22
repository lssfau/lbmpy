from lbmpy.velocityspace import LBStencilBase

import sympy as sp


class GaussHermiteQuadratureStencil(LBStencilBase):
    """Base class for Gauss-Hermite quadrature stencils."""

    def __new__(cls, *args, **kwargs):
        if cls is GaussHermiteQuadratureStencil:
            raise TypeError("Cannot create a generic Gauss-Hermite quadrature stencil")
        return super().__new__(cls)

    def __init__(
        self,
        velocities: tuple[tuple[int, ...], ...],
        ordering: str,
        stencil_name: str,
    ):
        self._check_valid_stencil(velocities)
        self._check_valid_ordering(ordering)

        weights = self.weights_per_energy_level(velocities)
        self._check_valid_weights(weights)

        super().__init__(
            velocities=velocities,
            theta=self.theta0,
            weights=weights,
            ordering=ordering,
            stencil_name=stencil_name,
        )


class D2V17(GaussHermiteQuadratureStencil):
    """Implementation of the D2V17 higher-order stencil.

    The velocity vectors of this stencil correspond to a seventh order Gauss-Hermite quadrature
    """

    orderings = dict(walberla=tuple(range(17)))
    _theta0 = sp.S(72 / (5 * (25 + sp.sqrt(193))))
    energy_based_weights = {
        0: sp.Rational(1, 8100) * (575 + 193 * sp.sqrt(193)),
        1: sp.Rational(1, 18000) * (3355 - 91 * sp.sqrt(193)),
        2: sp.Rational(1, 27000) * (655 + 17 * sp.sqrt(193)),
        8: sp.Rational(1, 54000) * (685 - 49 * sp.sqrt(193)),
        9: sp.Rational(1, 162000) * (1445 - 101 * sp.sqrt(193)),
    }

    def __init__(self, ordering: str = "walberla"):
        # fmt: off
        velocities = (
            (0, 0),                              # energy level = 0
            (0, -1), (-1, 0), (1, 0), (0, 1),    # energy level = 1
            (-1, -1), (1, -1), (-1, 1), (1, 1),  # energy level = 2
            (-2, -2), (2, -2), (-2, 2), (2, 2),  # energy level = 8
            (0, -3), (-3, 0), (3, 0), (0, 3),    # energy level = 9
        )

        # fmt:on

        super().__init__(
            velocities=velocities,
            ordering=ordering,
            stencil_name="D2V17",
        )


class D2V37(GaussHermiteQuadratureStencil):
    """Implementation of the D2V37 higher-order stencil.

    The velocity vectors of this stencil correspond to a ninth order Gauss-Hermite quadrature
    """

    orderings = dict(walberla=tuple(range(37)))
    _theta0 = sp.S(f"{(1/1.19697977039307435897239)**2}")
    energy_based_weights = {
        0: sp.S(0.23315066913235250228650),
        1: sp.S(0.10730609154221900241246),
        2: sp.S(0.05766785988879488203006),
        4: sp.S(0.01420821615845075026469),
        5: sp.S(0.00535304900051377523273),
        8: sp.S(0.00101193759267357547541),
        9: sp.S(0.00024530102775771734547),
        10: sp.S(0.00028341425299419821740),
    }

    def __init__(self, ordering: str = "walberla"):
        # fmt: off
        velocities = (
            (0, 0),                                # energy level = 0
            (0, -1), (-1, 0), (1, 0), (0, 1),      # energy level = 1
            (-1, -1), (1, -1), (-1, 1), (1, 1),    # energy level = 2
            (0, -2), (-2, 0), (2, 0), (0, 2),      # energy level = 4
            (-1, -2), (1, -2), (-2, -1), (2, -1),  # energy level = 5
            (-2, 1), (2, 1), (-1, 2), (1, 2),      # energy level = 5
            (-2, -2), (2, -2), (-2, 2), (2, 2),    # energy level = 8
            (0, -3), (-3, 0), (3, 0), (0, 3),      # energy level = 9
            (-1, -3), (1, -3), (-3, -1), (3, -1),  # energy level = 10
            (-3, 1), (3, 1), (-1, 3), (1, 3),      # energy level = 10
        )

        # fmt:on
        super().__init__(
            velocities=velocities,
            ordering=ordering,
            stencil_name="D2V37",
        )
