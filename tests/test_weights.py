import pytest
from lbmpy.creationfunctions import create_lb_method, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.stencils import LBStencil
import sympy as sp

def get_weights(stencil):
    def weight_for_direction(direction):
        squared_length = sum([d**2 for d in direction])
        return get_weights.weights[stencil.D][stencil.Q][squared_length]

    return [weight_for_direction(d) for d in stencil]

# fmt:off
get_weights.weights = dict({
    2: dict({
        9: dict({
            0: sp.Rational(4, 9), 1: sp.Rational(1, 9), 2: sp.Rational(1, 36),
        }),
        # weights taken from Coreixas et al. (2017), PRE.
        # https://doi.org/10.1103/PhysRevE.96.033306
        # (Appendix D Table 1)
        17: dict({
            0: sp.S((575 + 193 * sp.sqrt(193)) / 8100), 1: sp.S((3355 - 91 * sp.sqrt(193)) / 18000),
            2: sp.S((655 + 17 * sp.sqrt(193)) / 27000), 8: sp.S((685 - 49 * sp.sqrt(193)) / 54000),
            9: sp.S((1445 - 101 * sp.sqrt(193)) / 162000),
        }),
        # weights taken from Coreixas et al. (2017), PRE.
        # https://doi.org/10.1103/PhysRevE.96.033306
        # (Appendix D Table 1)
        37: dict({
            0: sp.S(0.23315066913235250228650), 1: sp.S(0.10730609154221900241246),
            2: sp.S(0.05766785988879488203006), 4: sp.S(0.01420821615845075026469),
            5: sp.S(0.00535304900051377523273), 8: sp.S(0.00101193759267357547541),
            9: sp.S(0.00024530102775771734547), 10: sp.S(0.00028341425299419821740),
        }),
    }),
    3: dict({
        7: dict({
            0: sp.core.numbers.Zero(), 1: sp.Rational(1, 6),
        }),
        15: dict({
            0: sp.Rational(2, 9), 1: sp.Rational(1, 9), 3: sp.Rational(1, 72),
        }),
        19: dict({
            0: sp.Rational(1, 3), 1: sp.Rational(1, 18), 2: sp.Rational(1, 36),
        }),
        27: dict({
            0: sp.Rational(8, 27), 1: sp.Rational(2, 27),
            2: sp.Rational(1, 54), 3: sp.Rational(1, 216),
        }),
    }),
})

def compare_weights(method, zero_centered, continuous_equilibrium, stencil_name):
    stencil = LBStencil(stencil_name)
    hardcoded_weights = get_weights(stencil)

    method = create_lb_method(LBMConfig(stencil=stencil, method=method, zero_centered=zero_centered, 
                                        continuous_equilibrium=continuous_equilibrium))
    weights = method.weights

    for i in range(len(weights)):
        assert hardcoded_weights[i] == weights[i]


@pytest.mark.parametrize('method', [Method.SRT, Method.TRT])
@pytest.mark.parametrize('zero_centered', [False, True])
@pytest.mark.parametrize('continuous_equilibrium', [False, True])
@pytest.mark.parametrize('stencil_name', [Stencil.D2Q9, Stencil.D3Q7])
def test_weight_calculation(method, zero_centered, continuous_equilibrium, stencil_name):
    compare_weights(method, zero_centered, continuous_equilibrium, stencil_name)


@pytest.mark.parametrize('method', [Method.MRT, Method.CENTRAL_MOMENT])
@pytest.mark.parametrize('continuous_equilibrium', [False, True])
@pytest.mark.parametrize('zero_centered', [False, True])
@pytest.mark.parametrize('stencil_name', [Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.longrun
def test_weight_calculation_longrun(method, zero_centered, continuous_equilibrium, stencil_name):
    compare_weights(method, zero_centered, continuous_equilibrium, stencil_name)