from pystencils import __version__ as ps_version

#   Determine if we're running pystencils 1.x or 2.x
version_tokes = ps_version.split(".")

PYSTENCILS_VERSION_MAJOR = int(version_tokes[0])
IS_PYSTENCILS_2 = PYSTENCILS_VERSION_MAJOR == 2

if IS_PYSTENCILS_2:
    from pystencils.defaults import DEFAULTS

    def get_loop_counter_symbol(coord: int):
        return DEFAULTS.spatial_counters[coord]

    def get_supported_instruction_sets():
        from pystencils import Target
        vector_targets = Target.available_vector_cpu_targets()
        isas = []
        for target in vector_targets:
            tokens = target.name.split("_")
            isas.append(tokens[-1].lower())
        return isas

else:
    from pystencils.backends.simd_instruction_sets import (
        get_supported_instruction_sets as get_supported_instruction_sets_,
    )

    get_supported_instruction_sets = get_supported_instruction_sets_

    def get_loop_counter_symbol(coord: int):
        from pystencils.astnodes import LoopOverCoordinate

        return LoopOverCoordinate.get_loop_counter_symbol(coord)


def import_guard_pystencils1(feature):
    if IS_PYSTENCILS_2:
        raise ImportError(
            f"The following feature is not yet available when running pystencils 2.x: {feature}"
        )
    return True
