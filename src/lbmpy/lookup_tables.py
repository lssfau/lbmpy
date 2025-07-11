from typing import Sequence, Any
from abc import ABC, abstractmethod
import numpy as np
import sympy as sp

from ._compat import IS_PYSTENCILS_2

if not IS_PYSTENCILS_2:
    raise ImportError("`lbmpy.lookup_tables` is only available when running with pystencils 2.x")

from pystencils import Assignment
from pystencils.sympyextensions import TypedSymbol
from pystencils.types.quick import Arr
from pystencils.types import UserTypeSpec, create_type


class LookupTables(ABC):
    @abstractmethod
    def get_array_declarations(self) -> list[Assignment]:
        pass


class NeighbourOffsetArrays(LookupTables):

    @staticmethod
    def neighbour_offset(dir_idx, stencil):
        if isinstance(sp.sympify(dir_idx), sp.Integer):
            return stencil[dir_idx]
        else:
            return tuple(
                [
                    sp.IndexedBase(symbol, shape=(1,))[dir_idx]
                    for symbol in NeighbourOffsetArrays._offset_symbols(stencil)
                ]
            )

    @staticmethod
    def _offset_symbols(stencil):
        q = len(stencil)
        dim = len(stencil[0])
        return [
            TypedSymbol(f"neighbour_offset_{d}", Arr(create_type("int32"), q))
            for d in ["x", "y", "z"][:dim]
        ]

    def __init__(self, stencil, offsets_dtype: UserTypeSpec = np.int32):
        self._offsets_dtype = create_type(
            offsets_dtype
        )  # TODO: Currently, this has no effect
        self._stencil = stencil
        self._dim = len(stencil[0])

    def get_array_declarations(self) -> list[Assignment]:
        array_symbols = NeighbourOffsetArrays._offset_symbols(self._stencil)
        return [
            Assignment(arrsymb, tuple((d[i] for d in self._stencil)))
            for i, arrsymb in enumerate(array_symbols)
        ]


class MirroredStencilDirections(LookupTables):

    @staticmethod
    def mirror_stencil(direction, mirror_axis):
        assert mirror_axis <= len(
            direction
        ), f"only {len(direction)} axis available for mirage"
        direction = list(direction)
        direction[mirror_axis] = -direction[mirror_axis]

        return tuple(direction)

    @staticmethod
    def _mirrored_symbol(mirror_axis, stencil):
        axis = ["x", "y", "z"]
        q = len(stencil)
        return TypedSymbol(
            f"{axis[mirror_axis]}_axis_mirrored_stencil_dir", Arr(create_type("int32"), q)
        )

    def __init__(self, stencil, mirror_axis, dtype=np.int32):
        self._offsets_dtype = create_type(dtype)  # TODO: Currently, this has no effect

        self._mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(
            mirror_axis, stencil
        )
        self._mirrored_directions = tuple(
            stencil.index(
                MirroredStencilDirections.mirror_stencil(direction, mirror_axis)
            )
            for direction in stencil
        )

    def get_array_declarations(self) -> list[Assignment]:
        return [Assignment(self._mirrored_stencil_symbol, self._mirrored_directions)]


class LbmWeightInfo(LookupTables):
    def __init__(self, lb_method, data_type="double"):
        self._weights = lb_method.weights
        self._weights_array = TypedSymbol("weights", Arr(create_type(data_type), len(self._weights)))

    def weight_of_direction(self, dir_idx, lb_method=None):
        if isinstance(sp.sympify(dir_idx), sp.Integer):
            assert lb_method is not None
            return lb_method.weights[dir_idx].evalf(17)
        else:
            return sp.IndexedBase(self._weights_array, shape=(1,))[dir_idx]

    def get_array_declarations(self) -> list[Assignment]:
        return [Assignment(self._weights_array, tuple(self._weights))]


class TranslationArraysNode(LookupTables):

    def __init__(self, array_content: Sequence[tuple[TypedSymbol, Sequence[Any]]]):
        self._decls = [
            Assignment(symb, tuple(content)) for symb, content in array_content
        ]

    def __str__(self):
        return "Variable PDF Access Translation Arrays"

    def __repr__(self):
        return "Variable PDF Access Translation Arrays"

    def get_array_declarations(self) -> list[Assignment]:
        return self._decls
