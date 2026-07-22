from lbmpy.velocityspace import LBStencilBase

import sympy as sp


class StandardStencil(LBStencilBase):
    r"""
    Base class for first nearest neighbour stencils.
    """

    _generating_velocities = list([0, -1, 1])
    _theta0 = sp.Rational(1, 3)

    def __new__(cls, *args, **kwargs):
        if cls is StandardStencil:
            raise TypeError("Cannot create a generic standard stencil.")
        return super().__new__(cls)

    def __init__(
        self,
        velocities: tuple[tuple[int, ...], ...],
        ordering: str,
        stencil_name: str,
        theta: sp.Expr,
    ):
        self._check_valid_stencil(velocities)
        self._check_valid_ordering(ordering)

        ordered_velocities = tuple(velocities[i] for i in self.orderings[ordering])

        weights = self.weights_per_energy_level(ordered_velocities)
        self._check_valid_weights(weights)

        super().__init__(
            velocities=ordered_velocities,
            theta=theta,
            weights=weights,
            ordering=ordering,
            stencil_name=stencil_name,
        )

    @property
    def generating_velocities(self):
        """Cardinal one-dimensional velocities"""
        return self._generating_velocities

    @property
    def Q(self):
        """Conventional notation for `num_velocities`"""
        return self.num_velocities


class D2Q9(StandardStencil):
    r"""
    Implementation of the D2Q9 standard stencil.

    Args:
        ordering:
            layout of the stencil entries.
            Currently supported orderings: walberla, counterclockwise, braunschweig, uk, lehmann
        temperature:
            Lattice temperature (default :math:`\theta_0` = 1/3)
    """

    orderings = dict(
        walberla=tuple([0, 2, 1, 3, 6, 5, 8, 4, 7]),
        counterclockwise=tuple([0, 6, 2, 3, 1, 8, 5, 4, 7]),
        braunschweig=tuple([0, 5, 3, 4, 1, 7, 6, 8, 2]),
        uk=tuple([0, 6, 3, 2, 1, 8, 4, 5, 7]),
        lehmann=tuple([0, 6, 3, 2, 1, 8, 4, 7, 5]),
    )

    def __init__(self, ordering: str = "walberla", temperature: sp.Expr | None = None):
        velocities = tuple(
            [
                (x, y)
                for x in self.generating_velocities
                for y in self.generating_velocities
            ]
        )
        theta = self.theta0 if temperature is None else temperature
        # W(theta) obtained as tensor product of D1Q3 weights which are obtained by solving
        # \sum wi = 1, \sum c_{i_\alpha}^2 wi = \theta.
        # (Eq2.8) Singh et al. (2013), Communications in Computational Physics. 2013;13(3):603-613.
        self.energy_based_weights = {
            0: ((1 - theta) ** 2),
            1: (sp.Rational(theta, 2) * (1 - theta)),
            2: (sp.Rational(theta, 2) ** 2),
        }
        super().__init__(velocities, ordering, "D2Q9", theta)


class D3Q7(StandardStencil):
    r"""
    Implementation of the D3Q7 standard stencil.

    Args:
        ordering:
            layout of the stencil entries.
            Currently supported orderings: walberla
        temperature:
            Lattice temperature (default :math:`\theta_0` = 1/3)
    """

    orderings = dict(walberla=tuple(range(7)))

    def __init__(self, ordering: str = "walberla", temperature: sp.Expr | None = None):
        # fmt:off
        velocities = (
            (0, 0, 0), (0, 1, 0), (0, -1, 0), (-1, 0, 0),
            (1, 0, 0), (0, 0, 1), (0, 0, -1),
        )

        # fmt:on
        theta = self.theta0 if temperature is None else temperature
        # W(theta) derived by solving \sum w_i = 1, \sum w_i c_{i_alpha}^2 = \theta
        self.energy_based_weights = {
            0: (1 - 3 * theta),
            1: sp.Rational(theta, 2),
        }
        super().__init__(velocities, ordering, "D3Q7", theta)


class D3Q27(StandardStencil):
    r"""
    Implementation of the D3Q27 standard stencil.

    Args:
        ordering:
            layout of the stencil entries.
            Currently supported orderings: walberla, premnath, fakhari, lehmann, braunschweig
        temperature:
            Lattice temperature (default :math:`\theta_0` = 1/3)
    """

    # fmt:off
    orderings = dict(
        walberla=tuple(
            [0, 6, 3, 9, 18, 2, 1, 15, 24, 12, 21, 8, 5, 11, 20, 7, 4, 10, 19, 26, 17, 23, 14, 25, 16, 22, 13,]
        ),
        premnath=tuple(
            [0, 18, 9, 6, 3, 2, 1, 24, 15, 21, 12, 20, 11, 19, 10, 8, 5, 7, 4, 26, 17, 23, 14, 25, 16, 22, 13,]
        ),
        fakhari=tuple(
            [0, 18, 9, 6, 3, 2, 1, 26, 17, 23, 14, 25, 16, 22, 13, 24, 15, 21, 12, 20, 11, 19, 10, 8, 5, 7, 4,]
        ),
        lehmann=tuple(
            [0, 18, 9, 6, 3, 2, 1, 24, 12, 20, 10, 8, 4, 21, 15, 19, 11, 7, 5, 26, 13, 25, 14, 23, 16, 17, 22,]
        ),
        braunschweig=tuple(
            [0, 18, 9, 6, 3, 2, 1, 24, 12, 21, 15, 20, 10, 19, 11, 8, 4, 7, 5, 26, 17, 23, 14, 25, 16, 22, 13,]
        )
    )

    # fmt:on
    def __init__(self, ordering: str = "walberla", temperature: sp.Expr | None = None):
        velocities = tuple(
            [
                (x, y, z)
                for x in self.generating_velocities
                for y in self.generating_velocities
                for z in self.generating_velocities
            ]
        )
        theta = self.theta0 if temperature is None else temperature
        # W(theta) obtained as tensor product of D1Q3 weights which are obtained by solving
        # \sum wi = 1, \sum c_{i_\alpha}^2 wi = \theta.
        # (Eq2.8) Singh et al. (2013), Communications in Computational Physics. 2013;13(3):603-613.
        self.energy_based_weights = {
            0: ((1 - theta) ** 3),
            1: (sp.Rational(theta, 2) * (1 - theta) ** 2),
            2: ((1 - theta) * sp.Rational(theta, 2) ** 2),
            3: (sp.Rational(theta, 2) ** 3),
        }
        super().__init__(velocities, ordering, "D3Q27", theta)


class D3Q19(StandardStencil):
    r"""
    Implementation of the D3Q19 standard stencil.

    Args:
        ordering:
            layout of the stencil entries.
            Currently supported orderings: walberla, counterclockwise, premnath, lehmann, braunschweig
        temperature:
            Lattice temperature (default :math:`\theta_0` = 1/3)
    """

    # fmt:off
    orderings = dict(
        walberla=tuple(
            [0, 6, 3, 9, 14, 2, 1, 13, 18, 12, 17, 8, 5, 11, 16, 7, 4, 10, 15]
        ),
        counterclockwise=tuple(
            [0, 14, 9, 6, 3, 2, 1, 18, 12, 16, 10, 8, 4, 17, 13, 15, 11, 7, 5]
        ),
        premnath=tuple(
            [0, 14, 9, 6, 3, 2, 1, 18, 13, 17, 12, 16, 11, 15, 10, 8, 5, 7, 4]
        ),
        lehmann=tuple(
            [0, 14, 9, 6, 3, 2, 1, 18, 12, 16, 10, 8, 4, 17, 13, 15, 11, 7, 5]
        ),
        braunschweig=tuple(
            [0, 14, 9, 6, 3, 2, 1, 18, 12, 17, 13, 16, 10, 15, 11, 8, 4, 7, 5]
        ),
    )

    # fmt:on
    def __init__(self, ordering: str = "walberla", temperature: sp.Expr | None = None):
        c27 = list(
            [
                (x, y, z)
                for x in self.generating_velocities
                for y in self.generating_velocities
                for z in self.generating_velocities
            ]
        )
        # fmt:off
        exclude = {
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
        }
        # fmt:on
        velocities = tuple([ci for ci in c27 if ci not in exclude])
        theta = self.theta0 if temperature is None else temperature
        # W(theta) is obtained by solving (see eq 3.60 Kruger et al. (2016) Springer International Publishing)
        # \sum_i w_i = 1 => 2w0 + 6w1 + 12w2 = 1 (1)
        # \sum_i w_i c_{i_\alpha} = \theta => 2w1 + 8w2 = \theta (2)
        # \sum_i w_i c_{i_\alpha}^4 = \theta => 2w1 + 8w2 = 3\theta (3)
        # \sum_i w_i c_{i_\alpha}^2c_{i_\beta}^2 = \theta^2 => 4w2 = \theta^2 (4)
        # at \theta \neq 1/3 (3) is not satisfied
        self.energy_based_weights = {
            0: (1 - 3 * theta + 3 * theta**2),
            1: (sp.Rational(theta, 2) * (1 - 2 * theta)),
            2: sp.Rational(theta**2, 4),
        }
        super().__init__(velocities, ordering, "D3Q19", theta)


class D3Q15(StandardStencil):
    r"""
    Implementation of the D3Q19 standard stencil.

    Args:
        ordering:
            layout of the stencil entries.
            Currently supported orderings: walberla, premnath, lehmann, fakhari
        theta:
            Lattice temperature (default theta_0 = 1/3)
    """

    # fmt:off
    orderings = dict(
        walberla=tuple([0, 4, 3, 5, 10, 2, 1, 14, 9, 12, 7, 13, 8, 11, 6]),
        premnath=tuple([0, 10, 5, 4, 3, 2, 1, 14, 9, 12, 7, 13, 8, 11, 6]),
        lehmann=tuple([0, 10, 5, 4, 3, 2, 1, 14, 6, 13, 7, 12, 8, 9, 11]),
        fakhari=tuple([0, 10, 5, 4, 3, 2, 1, 14, 6, 9, 11, 12, 8, 13, 7]),
    )

    # fmt:on
    def __init__(self, ordering: str = "walberla", temperature: sp.Expr | None = None):
        c27 = list(
            [
                (x, y, z)
                for x in self.generating_velocities
                for y in self.generating_velocities
                for z in self.generating_velocities
            ]
        )
        # fmt:off
        exclude = {
            (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1), (-1, 0, -1), (-1, 0, 1),
            (-1, -1, 0), (-1, 1, 0), (1, 0, -1), (1, 0, 1), (1, -1, 0), (1, 1, 0),
        }
        # fmt:on
        velocities = tuple([ci for ci in c27 if ci not in exclude])
        theta = self.theta0 if temperature is None else temperature
        # W(theta) is obtained by solving (see eq 3.60 Kruger et al. (2016) Springer International Publishing)
        # \sum_i w_i = 1 => w0 + 6w1 + 8w2 = 1 (1)
        # \sum_i w_i c_{i_\alpha} = \theta => 2w1 + 8w2 = \theta (2)
        # \sum_i w_i c_{i_\alpha}^4 = 3\theta => 2w1 + 8w2 = 3\theta^2 (3)
        # \sum_i w_i c_{i_\alpha}^2c_{i_\beta}^2 = \theta^2 => 8w3 = \theta^2 (4)
        # at \theta \neq 1/3 (3) is not satisfied
        self.energy_based_weights = {
            0: (1 - theta) * (1 - 2 * theta),
            1: sp.Rational(theta, 2) * (1 - theta),
            3: sp.Rational(1, 8) * theta**2,
        }
        super().__init__(velocities, ordering, "D3Q15", theta)
