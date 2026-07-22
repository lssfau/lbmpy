from __future__ import annotations

import pystencils as ps

import sympy as sp

from enum import Enum, auto


class StencilType(Enum):
    """The StencilType enumeration indicates the type of stencil employed for LB simulations."""

    OnLattice = auto()
    """On-lattice stencils have velocities terminating on nodes allowing for traditional streaming.
    """
    OffLattice = auto()
    """
    Off-lattice stencils terminate between nodes yielding semi-lagrangian LB schemes in which data
    needs to be interpolated on to the nodes followed by streaming.
    Currently semi-lagrangian schemes are not supported.
    """


class LBStencilBase:
    r"""
    Base Class for lattice Boltzmann stencils.

    Args:
        velocities
            The ordered velocities associated with the stencil.
        theta:
            Specifies the lattice temperature.
            For athermal simulations, :math:`\theta = \theta_0`, i.e., the lattice reference temperature or,
            equivalently, squared lattice speed of sound (:math:`c_s^2`). For isothermal simulations, :math:`\theta` is
            a constant, while for thermal simulations, :math:`\theta` is a dynamically varying sympy symbol
        weights:
            tuple of sympy expressions that describe the stencil weights for ordered stencil velocities
        ordering:
            Layout of the stencil velocities.
            The ordering argument allows for comparisons against published results that employ different layouts for
            the same stencil.
        stencil_name:
            Name of the stencil
        stencil_type:
            type of stencil, namely, on-lattice or off-lattice.
            On-lattice stencils have velocities terminating on nodes allowing for traditional  streaming, while
            off-lattice stencils terminate between nodes yielding semi-lagrangian LB schemes.
            Currently, streaming is possible only for on-lattice stencils.
    """

    orderings: dict[str, tuple[int, ...]]
    _theta0: sp.Expr
    energy_based_weights: dict[int, sp.Expr]

    def __new__(cls, *args, **kwargs):
        if cls is LBStencilBase:
            raise TypeError("Cannot create a generic base stencil.")
        return super().__new__(cls)

    def __init__(
        self,
        velocities: tuple[tuple[int, ...], ...],
        theta: sp.Expr | None = None,
        *,
        weights: tuple[sp.Expr, ...] | None = None,
        ordering: str | None = None,
        stencil_name: str | None = None,
        stencil_type: StencilType = StencilType.OnLattice,
    ):
        self._stencil_entries = velocities
        self._theta = theta
        self._weights = weights
        self._ordering = ordering
        self._stencil_name = stencil_name
        self._stencil_type = stencil_type
        self._q = len(velocities)
        self._dim = len(velocities[0])

        if len(self.weights) != self.num_velocities:
            raise ValueError(f"weights must have {self.num_velocities} elements.")

    def _check_valid_stencil(self, velocities: tuple[tuple[int, ...], ...]):
        valid_stencil = ps.stencil.is_valid(velocities)
        if not valid_stencil:
            raise ValueError(
                "The stencil you have created is not valid. "
                "It probably contains elements with different lengths."
            )

        if len(set(velocities)) < len(velocities):
            raise ValueError(
                "The stencil you have created is not valid: "
                "it contains duplicate elements"
            )

    def _check_valid_ordering(self, ordering: str):
        if len(self.orderings) > 0 and ordering not in self.orderings:
            err_message = ", ".join(self.orderings)
            raise ValueError(
                f"Unknown ordering: {ordering}\n"
                f"Supported orderings are: {err_message}"
            )

    def _check_valid_weights(self, weights):
        if sum(weights).nsimplify() != sp.core.numbers.One():
            raise ValueError("Invalid weights. Weights must add up to 1.")

    def weights_per_energy_level(self, velocities):
        if not hasattr(self, "energy_based_weights"):
            raise ValueError("Missing information: energy_based_weights") from None

        def squared_length(direction):
            return sum([d**2 for d in direction])

        try:
            return tuple(
                [self.energy_based_weights[squared_length(d)] for d in velocities]
            )
        except KeyError:
            raise ValueError(
                "The energy_based_weights dict must have an entry for each energy level."
            )

    @property
    def D(self):
        """Stencil dimension."""
        return self._dim

    @property
    def num_velocities(self):
        """Number of stencil velocities."""
        return self._q

    @property
    def theta(self):
        """Stencil operating temperature."""
        return self._theta

    @property
    def theta0(self):
        """Stencil reference temperature."""
        return self._theta0

    @property
    def speed_of_sound(self):
        """Stencil speed of sound."""
        return sp.sqrt(self.theta0)

    @property
    def stencil_type(self):
        """Stencil stencil type."""
        return self._stencil_type

    @property
    def ordering(self):
        """Stencil ordering."""
        return self._ordering

    @property
    def name(self) -> str | None:
        """Name of the stencil."""
        return self._stencil_name.upper()

    @property
    def stencil_entries(self):
        """Stencil velocity entries."""
        return self._stencil_entries

    @property
    def weights(self) -> tuple[sp.Expr, ...]:
        """Stencil weights."""
        return self._weights

    @property
    def inverse_stencil_entries(self):
        """Inverse Stencil entries."""
        return tuple([ps.stencil.inverse_direction(d) for d in self._stencil_entries])

    def plot(self, slice=False, **kwargs):
        """Create a plot of the stencil."""
        ps.stencil.plot(stencil=self._stencil_entries, slice=slice, **kwargs)

    def index(self, direction):
        """Index of a stencil velocity in the specified direction."""
        if len(direction) != self.D:
            raise ValueError("direction must match stencil.D")
        return self._stencil_entries.index(direction)

    def inverse_index(self, direction):
        """Index of stencil velocity entry opposite to the specified direction."""
        if len(direction) != self.D:
            raise ValueError("direction must match stencil.D")
        direction = ps.stencil.inverse_direction(direction)
        return self._stencil_entries.index(direction)

    def __getitem__(self, index):
        return self._stencil_entries[index]

    def __iter__(self):
        yield from self._stencil_entries

    def _args(self) -> tuple:
        return (
            self.ordering,
            self.stencil_entries,
            self.weights,
            self.stencil_type,
            self.name,
            self.theta,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LBStencilBase):
            return False
        return self._args() == other._args()

    def __len__(self):
        return len(self._stencil_entries)

    def __str__(self):
        return str(self._stencil_entries)

    def __hash__(self):
        return hash((type(self),) + self._args())

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Nr.</th>
                <th {nb} >Direction Name</th>
                <th {nb} >Direction </th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for i, direction in enumerate(self._stencil_entries):
            vals = {
                "nr": sp.latex(i),
                "name": sp.latex(ps.stencil.offset_to_direction_string(direction)),
                "entry": sp.latex(direction),
                "nb": 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>${nr}$</td>
                            <td {nb}>${name}$</td>
                            <td {nb}>${entry}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')
