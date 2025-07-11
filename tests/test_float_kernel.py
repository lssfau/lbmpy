import pytest

import pystencils as ps

from lbmpy.creationfunctions import create_lb_function, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.stencils import LBStencil

from lbmpy._compat import IS_PYSTENCILS_2


@pytest.mark.parametrize("double_precision", [False, True])
@pytest.mark.parametrize(
    "method_enum", [Method.SRT, Method.CENTRAL_MOMENT, Method.CUMULANT]
)
def test_creation(double_precision, method_enum):
    """Simple test that makes sure that only float variables are created"""
    lbm_config = LBMConfig(method=method_enum, relaxation_rate=1.5, compressible=True)
    if IS_PYSTENCILS_2:
        config = ps.CreateKernelConfig(
            default_dtype="float64" if double_precision else "float32"
        )
    else:
        config = ps.CreateKernelConfig(
            data_type="float64" if double_precision else "float32",
            default_number_float="float64" if double_precision else "float32",
        )
    func = create_lb_function(lbm_config=lbm_config, config=config)
    code = ps.get_code_str(func)

    if double_precision:
        assert "float" not in code
        assert "double" in code
    else:
        assert "double" not in code
        assert "float" in code


@pytest.mark.parametrize("numeric_type", ["float32", "float64"])
@pytest.mark.parametrize(
    "method_enum", [Method.SRT, Method.CENTRAL_MOMENT, Method.CUMULANT]
)
def test_scenario(method_enum, numeric_type):
    lbm_config = LBMConfig(
        stencil=LBStencil(Stencil.D3Q27),
        method=method_enum,
        relaxation_rate=1.45,
        compressible=True,
    )

    if IS_PYSTENCILS_2:
        config = ps.CreateKernelConfig(
            default_dtype=numeric_type
        )
    else:
        config = ps.CreateKernelConfig(
            data_type=numeric_type,
            default_number_float=numeric_type
        )
    sc = create_lid_driven_cavity((16, 16, 8), lbm_config=lbm_config, config=config)
    sc.run(1)
    code = ps.get_code_str(sc.ast)

    if numeric_type == "float64":
        assert "float" not in code
        assert "double" in code
    else:
        assert "double" not in code
        assert "float" in code
