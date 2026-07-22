import warnings

import pytest
import sympy as sp
import matplotlib.pyplot as plt

import pystencils as ps
from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil
from lbmpy import velocityspace


def get_3d_stencils():
    return (
        LBStencil(Stencil.D3Q15),
        LBStencil(Stencil.D3Q19),
        LBStencil(Stencil.D3Q27),
        LBStencil(Stencil.D3Q7),
    )


def get_2d_stencils():
    return (
        LBStencil(Stencil.D2Q9),
        LBStencil(Stencil.D2V17),
        LBStencil(Stencil.D2V37),
    )


@pytest.mark.parametrize(
    "stencil, attribute, size",
    [
        pytest.param(Stencil.D2Q9, "Q", 9, id="D2Q9"),
        pytest.param(Stencil.D3Q7, "Q", 7, id="D3Q7"),
        pytest.param(Stencil.D3Q15, "Q", 15, id="D3Q15"),
        pytest.param(Stencil.D3Q19, "Q", 19, id="D3Q19"),
        pytest.param(Stencil.D3Q27, "Q", 27, id="D3Q27"),
        pytest.param(Stencil.D2V17, "num_velocities", 17, id="D2V17"),
        pytest.param(Stencil.D2V37, "num_velocities", 37, id="D2V37"),
    ],
)
def test_sizes(stencil, attribute, size):
    assert getattr(LBStencil(stencil), attribute) == size


@pytest.mark.parametrize(
    "stencil, dimension",
    [pytest.param(s, 3, id=s.name) for s in get_3d_stencils()]
    + [pytest.param(s, 2, id=s.name) for s in get_2d_stencils()],
)
def test_dimensionality(stencil, dimension):
    assert stencil.D == dimension
    assert all(len(entry) == dimension for entry in stencil.stencil_entries)


@pytest.mark.parametrize(
    "stencil",
    [pytest.param(s, id=s.name) for s in get_3d_stencils()]
    + [pytest.param(s, id=s.name) for s in get_2d_stencils()],
)
def test_uniqueness(stencil):
    direction_set = set(stencil.stencil_entries)
    assert len(direction_set) == len(stencil.stencil_entries)


@pytest.mark.parametrize(
    "stencil, ordering, neighborhood",
    [
        pytest.param(LBStencil(Stencil.D2Q9), ordering, 1, id=f"d2q9-{ordering}")
        for ordering in velocityspace.D2Q9.orderings
    ]
    + [
        pytest.param(LBStencil(Stencil.D3Q7), ordering, 1, id=f"d3q7-{ordering}")
        for ordering in velocityspace.D3Q7.orderings
    ]
    + [
        pytest.param(LBStencil(Stencil.D3Q15), ordering, 1, id=f"d3q15-{ordering}")
        for ordering in velocityspace.D3Q15.orderings
    ]
    + [
        pytest.param(LBStencil(Stencil.D3Q19), ordering, 1, id=f"d3q19-{ordering}")
        for ordering in velocityspace.D3Q19.orderings
    ]
    + [
        pytest.param(LBStencil(Stencil.D3Q27), ordering, 1, id=f"d3q27-{ordering}")
        for ordering in velocityspace.D3Q27.orderings
    ]
    + [
        pytest.param(LBStencil(Stencil.D2V17), ordering, 3, id=f"d2v17-{ordering}")
        for ordering in velocityspace.D2V17.orderings
    ]
    + [
        pytest.param(LBStencil(Stencil.D2V37), ordering, 3, id=f"d2v37-{ordering}")
        for ordering in velocityspace.D2V17.orderings
    ],
)
def test_run_self_check(stencil, ordering, neighborhood):
    assert ps.stencil.is_valid(stencil.stencil_entries, max_neighborhood=neighborhood)
    assert ps.stencil.is_symmetric(stencil.stencil_entries)


@pytest.mark.parametrize(
    "stencil",
    [pytest.param(s, id=s.name) for s in get_2d_stencils()]
    + [pytest.param(s, id=s.name) for s in get_3d_stencils()],
)
def test_inverse_direction(stencil):
    assert all(
        ps.stencil.inverse_direction(stencil[i]) == stencil.inverse_stencil_entries[i]
        for i in range(stencil.num_velocities)
    )


def test_stencil_index():
    stencil = LBStencil(Stencil.D2Q9)
    assert stencil.index((1, 0)) == 4
    with pytest.raises(ValueError):
        _ = stencil.index((1, 0, 1))


def test_stencil_inverse_index():
    stencil = LBStencil(Stencil.D2Q9)
    assert stencil.inverse_index((1, 0)) == 3
    with pytest.raises(ValueError):
        _ = stencil.inverse_index((1, 0, 1))


@pytest.mark.parametrize(
    "stencil",
    [pytest.param(s, id=s.name) for s in get_2d_stencils()]
    + [pytest.param(s, id=s.name) for s in get_3d_stencils()],
)
def test_stencil_speed_of_sound(stencil):
    assert stencil.speed_of_sound**2 == stencil.theta0


@pytest.mark.parametrize(
    "stencil",
    [pytest.param(s, id=s.name) for s in get_2d_stencils()]
    + [pytest.param(s, id=s.name) for s in get_3d_stencils()],
)
def test_stencil_length(stencil):
    assert len(stencil) == stencil.num_velocities


def test_stencil_str_repr():
    stencil = LBStencil(
        ((-1,), (0,), (1,)),
        theta0=sp.Rational(1, 3),
        weights=tuple([sp.Rational(1, 6), sp.Rational(2, 3), sp.Rational(1, 6)]),
        stencil_name="D1Q3",
        stencil_type=velocityspace.StencilType.OnLattice,
    )

    assert str(stencil) == "((-1,), (0,), (1,))"


def test_free_functions():
    assert not ps.stencil.is_symmetric([(1, 0), (0, 1)])
    assert not ps.stencil.is_valid([(1, 0), (1, 1, 0)])
    assert not ps.stencil.is_valid([(2, 0), (0, 1)], max_neighborhood=1)

    with pytest.raises(ValueError) as e:
        LBStencil("name_that_does_not_exist")
    assert (
        "Stencil 'name_that_does_not_exist' is not defined in lbmpy.enums.Stencil"
        in str(e.value)
    )


def test_visualize():
    plt.clf()
    plt.cla()

    d2q9, d3q19 = LBStencil(Stencil.D2Q9), LBStencil(Stencil.D3Q19)
    figure = plt.gcf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d2q9.plot(figure=figure, data=[str(i) for i in range(9)])
        d3q19.plot(figure=figure, data=sp.symbols("a_:19"))


def test_comparability_of_stencils():
    stencil_1 = LBStencil(Stencil.D2Q9)
    stencil_2 = LBStencil(Stencil.D2Q9)
    stencil_3 = LBStencil(Stencil.D2Q9, ordering="braunschweig")
    stencil_4 = LBStencil(
        stencil_1.stencil_entries,
        theta0=stencil_1.theta0,
        weights=stencil_1.weights,
        stencil_name="custom",
        stencil_type=stencil_1.stencil_type,
    )
    stencil_5 = LBStencil(
        stencil_3.stencil_entries,
        theta0=stencil_3.theta0,
        weights=stencil_3.weights,
        ordering=stencil_3.ordering,
        stencil_name=stencil_3.name,
        stencil_type=stencil_3.stencil_type,
    )
    stencil_6 = LBStencil(
        stencil_1.stencil_entries,
        theta0=stencil_1.theta0,
        weights=stencil_1.weights,
        stencil_name="custom",
        stencil_type=stencil_1.stencil_type,
    )

    assert stencil_1 == stencil_2
    assert stencil_1 != stencil_3
    assert stencil_1 != stencil_4
    assert stencil_1 != stencil_5
    assert stencil_4 == stencil_6
    assert stencil_1 != 5
    assert len({stencil_1, stencil_2}) == 1


@pytest.mark.parametrize(
    "lbstencil, ordering, stencil",
    [
        # fmt:off
        pytest.param(Stencil.D2Q9, "walberla", (
                (0, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1),
            ),
            id="D2Q9-walberla",
        ),
        pytest.param(Stencil.D2Q9, "counterclockwise", (
                (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1),
            ),
            id="D2Q9-counterclockwise",
        ),
        pytest.param(Stencil.D2Q9, "braunschweig", (
                (0, 0), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1),
            ),
            id="D2Q9-braunschweig",
        ),
        pytest.param(Stencil.D2Q9, "uk", (
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1),
            ),
            id="D2Q9-uk",
        ),
        pytest.param(Stencil.D2Q9, "lehmann", (
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1),
            ),
            id="D2Q9-lehmann",
        ),
        pytest.param(Stencil.D3Q7, "walberla", (
                (0, 0, 0), (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
            ),
            id="D3Q7-walberla",
        ),
        pytest.param(Stencil.D3Q15, "walberla", (
                (0, 0, 0), (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1),
                (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
            ),
            id="D3Q15-walberla",
        ),
        pytest.param(Stencil.D3Q15, "premnath", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1),
                (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
            ),
            id="D3Q15-premnath",
        ),
        pytest.param(Stencil.D3Q15, "lehmann", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1),
                (-1, 1, -1), (-1, 1, 1), (1, -1, -1),
            ),
            id="D3Q15-lehmann",
        ),
        pytest.param(Stencil.D3Q15, "fakhari", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 1), (-1, -1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1),
                (-1, 1, -1), (1, 1, -1), (-1, -1, 1),
            ),
            id="D3Q15-fakhari",
        ),
        pytest.param(Stencil.D3Q19, "walberla", (
                (0, 0, 0), (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1),
                (0, 0, -1), (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (0, 1, 1),
                (0, -1, 1), (-1, 0, 1), (1, 0, 1), (0, 1, -1), (0, -1, -1), (-1, 0, -1),
                (1, 0, -1),
            ),
            id="D3Q15-walberla",
        ),
        pytest.param(Stencil.D3Q19, "counterclockwise", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1), (0, 1, 1),
                (0, -1, -1), (1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, 1, -1),
                (0, -1, 1),
            ),
            id="D3Q19-counterclockwise",
        ),
        pytest.param(Stencil.D3Q19, "braunschweig", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1),
                (-1, 0, -1), (1, 0, -1), (-1, 0, 1), (0, 1, 1), (0, -1, -1), (0, 1, -1),
                (0, -1, 1),
            ),
            id="D3Q19-braunschweig",
        ),
        pytest.param(Stencil.D3Q19, "premnath", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1),
                (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
                (0, -1, -1),
            ),
            id="D3Q19-premnath"),
        pytest.param(Stencil.D3Q19, "lehmann", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1), (0, 1, 1),
                (0, -1, -1), (1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, 1, -1),
                (0, -1, 1),
             ),
            id="D3Q19-lehmann",
        ),
        pytest.param(Stencil.D3Q27, "walberla", (
                (0, 0, 0), (0, 1, 0),  (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1),
                (0, 0, -1), (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (0, 1, 1),
                (0, -1, 1), (-1, 0, 1), (1, 0, 1), (0, 1, -1), (0, -1, -1), (-1, 0, -1),
                (1, 0, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1),
                (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
            ),
            id="D3Q27-walberla",
        ),
        pytest.param(Stencil.D3Q27, "premnath", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1),
                (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
                (0, -1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1),
                (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
            ),
            id="D3Q27-premnath",
        ),
        pytest.param(Stencil.D3Q27, "fakhari", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1),
                (-1, 1, -1), (1, -1, -1), (-1, -1, -1), (1, 1, 0), (-1, 1, 0), (1, -1, 0),
                (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1),
                (0, -1, 1), (0, 1, -1), (0, -1, -1),
            ),
            id="D3Q27-fakhari",
        ),
        pytest.param(Stencil.D3Q27, "lehmann", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1), (0, 1, 1),
                (0, -1, -1), (1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, 1, -1),
                (0, -1, 1), (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1),
                (-1, 1, -1), (-1, 1, 1), (1, -1, -1),
            ),
            id="D3Q27-lehmann",
        ),
        pytest.param(Stencil.D3Q27, "braunschweig", (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                (0, 0, -1), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1),
                (-1, 0, -1), (1, 0, -1), (-1, 0, 1), (0, 1, 1), (0, -1, -1), (0, 1, -1),
                (0, -1, 1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1),
                (-1, 1, -1), (1, -1, -1),(-1, -1, -1),
            ),
            id="D3Q27-braunschweig",
        ),
        pytest.param(Stencil.D2V17, "walberla", (
                (0, 0), (0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1),
                (-2, -2), (2, -2), (-2, 2), (2, 2), (0, -3), (-3, 0), (3, 0), (0, 3),
            ),
            id="D2V17-walberla",
        ),
        pytest.param(Stencil.D2V37, "walberla", (
                (0, 0), (0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1),
                (0, -2), (-2, 0), (2, 0), (0, 2), (-1, -2), (1, -2), (-2, -1), (2, -1), (-2, 1),
                (2, 1), (-1, 2), (1, 2), (-2, -2), (2, -2), (-2, 2), (2, 2), (0, -3), (-3, 0),
                (3, 0), (0, 3), (-1, -3), (1, -3), (-3, -1), (3, -1), (-3, 1), (3, 1), (-1, 3),
                (1, 3),
            ),
            id="D2V37-walberla",
        )
        # fmt:on
    ],
)
def test_stencil_ordering(lbstencil, ordering, stencil):
    assert LBStencil(lbstencil, ordering=ordering).stencil_entries == stencil


@pytest.mark.parametrize(
    "stencil, kwargs",
    [
        pytest.param(
            [(-1,), (0,), (1,)],
            {
                "weights": (sp.Rational(1, 6), sp.Rational(2, 3), sp.Rational(1, 6)),
            },
            id="invalid-stencil-type",
        ),
        pytest.param(
            ((-1, 0), (0,), (1,)),
            {
                "weights": (sp.Rational(1, 6), sp.Rational(2, 3), sp.Rational(1, 6)),
            },
            id="invalid-velocity-entries",
        ),
        pytest.param(
            ((0,), (0,), (1,)),
            {
                "weights": (sp.Rational(1, 6), sp.Rational(2, 3), sp.Rational(1, 6)),
            },
            id="duplicate-velocity-entries",
        ),
        pytest.param(
            ((-1,), (0,), (1,)),
            {
                "weights": (sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(1, 6)),
            },
            id="non-normalized-weights",
        ),
        pytest.param(
            ((-1,), (0,), (1,)),
            {"weights": (sp.core.numbers.One(),)},
            id="insufficient-weights",
        ),
    ],
)
def test_stencil_error_branches(stencil, kwargs):
    with pytest.raises(ValueError):
        _ = LBStencil(
            stencil,
            theta0=sp.Rational(1, 3),
            stencil_name="D1Q3",
            stencil_type=velocityspace.StencilType.OnLattice,
            **kwargs,
        )


def test_stencil_missing_arguments():
    with pytest.raises(ValueError):
        _ = LBStencil(((-1,), (0,), (1,)))


def test_stencil_missing_enum_representation():
    with pytest.raises(ValueError):
        _ = LBStencil("invalid-stencil-type")


def test_missing_stencil_implementation(monkeypatch):
    monkeypatch.delattr(velocityspace, "D2Q9")
    with pytest.raises(NotImplementedError):
        _ = LBStencil(Stencil.D2Q9)


def test_invalid_stencil_ordering():
    with pytest.raises(ValueError):
        _ = LBStencil(Stencil.D2Q9, ordering="invalid-ordering")


def test_stencil_is_lbstencil():
    from lbmpy.lbstep import LatticeBoltzmannStep

    box = LatticeBoltzmannStep(
        stencil=LBStencil(Stencil.D3Q19),
        domain_size=(2, 2, 2),
        relaxation_rate=1.0,
        periodicity=(True, True, True),
    )
    assert box.method.stencil.name == "D3Q19"


def test_missing_stencil_weights_per_energy_level():
    from lbmpy.velocityspace.standardstencils import StandardStencil

    class D1Q3(StandardStencil):
        orderings = dict(walberla=tuple(range(3)))

        def __init__(self):
            super().__init__(((-1,), (0,), (1,)), "walberla", "D1Q3", None)

    with pytest.raises(ValueError):
        _ = D1Q3()

    D1Q3.energy_based_weights = {}

    with pytest.raises(ValueError):
        _ = D1Q3()


@pytest.mark.parametrize(
    "stencil",
    [
        pytest.param(velocityspace.LBStencilBase, id="LBStencilBase"),
        pytest.param(
            velocityspace.standardstencils.StandardStencil, id="StandardStencil"
        ),
        pytest.param(
            velocityspace.ghstencils.GaussHermiteQuadratureStencil,
            id="GaussHermiteQuadratureStencil",
        ),
    ],
)
def test_intantiate_stencil_base_classes(stencil):
    with pytest.raises(TypeError):
        _ = stencil()
