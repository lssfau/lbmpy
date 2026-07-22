****************************************************
Velocity Space Representations (lbmpy.velocityspace)
****************************************************

This module contains mechanisms for defining velocity space discretizations, commonly reffered to as stencils, needed
for lattice Boltzmann simulations.

The discrete stencil velocities can take integer/real values yielding on-/off-lattice stencils representations. On
uniform grids, velocities of on-lattice stencils terminate on adjacent grid nodes thereby allowing for the traditional
streaming operation; in contrast, the off-lattice stencil velocities terminate in between nodes and yield
semi-Lagrangian schemes where velocities must first be interpolated onto the grid nodes before streaming. *lbmpy*
allows users to declare if a stencil is `OnLattice` or `OffLattice` through the `StencilType` `enum`.

.. warning::
   *lbmpy* currently does not support semi-Lagrangian lattice Boltzmann schemes.


The charcteristics common to all *lbmpy* stencils are encapsulated in the `LBStencilBase` class, while all specific
realizations extend this base class. *lbmpy* currently provides the following stencil implementations:

- :ref:`standard_stencils`:

  The `StandardStencil` class extends `LBStencilBase` to encapsulate characteristics that are common to all standard,
  first-nearest-neighbour stencils. By extending `StandardStencil`, *lbmpy* provides users implementations of the
  :ref:`d2q9` and :ref:`d3Q27` tensor-product stencils, along with the :ref:`d3Q7`, :ref:`d3Q15` and :ref:`d3Q19`
  prunes of the :ref:`d3q27` stencil.

- :ref:`gauss_hermite_stencils`:

  Similar to `StandardStencil`, the `GaussHermiteQuadratureStencil` base class extends `LBStencilBase` and serves to
  encapsulate attributes common to stencils whose velocities are roots of Gauss-Hermite quadratures. Currently, *lbmpy*
  provides implementations of the 2D `StencilType.OnLattice` higher-order :ref:`d2v17` and
  :ref:`d2v37` stencils having 7\ :sup:`th` and 9\ :sup:`th` order accuracy respectively.

  .. warning::
     Higher-order stencils warrant additional considerations for boundary representations and the streaming operation.
     For end-to-end lattice Boltzmann simulations within the *lbmpy* framework, higher-order stencils can currently
     only be used on periodic domains. Additionally, the *waLBerla* backend does not currently support streaming
     patterns required for using higher-order stencils.

- :ref:`custom_stencil`:

  The `CustomStencil` class also extends `LBStencilBase` and allows users to define custom stencils for their
  applications.

  .. note::
     Incorrect inputs may yield physically incorrect/inconsistent results without raising errors; users are recommended
     to carefully verify their `CustomStencil` inputs.


Instantiation
=============

*lbmpy* stencils can be instantiated as given below.

- Using the :ref:`lb_stencil` method

  - Pre-defined stencils

    .. code-block:: python

        from lbmpy import LBStencil, Stencil

        # instantiate the D2Q9 stencil using lbmpy.enums.Stencil
        stencil_1 = LBStencil(Stencil.D2Q9)
        # instantiate the D2Q9 stencil using an upper-case string identifier
        stencil_2 = LBStencil("D2Q9")
        # instantiate the D2Q9 stencil using a lower-case string identifier
        stencil_3 = LBStencil("d2q9")

        assert(stencil_1 == stencil_2) # True
        assert(stencil_1 == stencil_3) # True



  - Custom stencils

    .. code-block:: python

        from lbmpy import LBStencil
        from lbmpy.velocityspace import StencilType

        stencil = LBStencil("d2q9")

        custom_stencil = LBStencil(
            stencil.stencil_entries,
            theta0=stencil.theta0,
            weights=stencil.weights,
            ordering="custom-ordering",
            stencil_name="custom-d2q9",
            stencil_type=StencilType.OnLattice,
        )

- Direct instantiation

  - Pre-defined stencils

    .. code-block:: python

        from lbmpy.velocityspace import D2Q9
        stencil = D2Q9()

  - Custom stencils

    .. code-block:: python

        from lbmpy import LBStencil
        from lbmpy.velocityspace import CustomStencil, StencilType

        stencil = LBStencil("d2q9")

        custom_stencil = CustomStencil(
            stencil.stencil_entries,
            theta0=stencil.theta0,
            weights=stencil.weights,
            ordering="custom-ordering",
            stencil_name="custom-d2q9",
            stencil_type=StencilType.OnLattice,
        )


.. _lb_stencil:

LBStencil Factory
=================

.. autoclass:: lbmpy.LBStencil
    :members:


.. _stencil_type:

StencilType
===========

.. autoclass:: lbmpy.velocityspace.StencilType
    :members:


.. _lb_stencil_base:

LBStencilBase
=============

.. autoclass:: lbmpy.velocityspace.LBStencilBase
    :members:



.. _standard_stencils:

Standard Stencils
=================

.. autoclass:: lbmpy.velocityspace.standardstencils.StandardStencil
    :members:

.. _d2q9:

D2Q9
----

.. autoclass:: lbmpy.velocityspace.D2Q9
    :members:

.. _d3q27:

D3Q27
-----

.. autoclass:: lbmpy.velocityspace.D3Q27
    :members:

.. _d3q7:

D3Q7
----

.. autoclass:: lbmpy.velocityspace.D3Q7
    :members:

.. _d3q15:

D3Q15
-----

.. autoclass:: lbmpy.velocityspace.D3Q15
    :members:

.. _d3q19:

D3Q19
-----

.. autoclass:: lbmpy.velocityspace.D3Q19
    :members:

.. _gauss_hermite_stencils:

Gauss Hermite Quadrature Stencils
=================================

.. autoclass:: lbmpy.velocityspace.ghstencils.GaussHermiteQuadratureStencil
    :members:

.. _d2v17:

D2V17
-----

.. autoclass:: lbmpy.velocityspace.D2V17
    :members:

.. _d2v37:

D2V37
-----

.. autoclass:: lbmpy.velocityspace.D2V37
    :members:

.. _custom_stencil:

User Defined Stencils
=====================

.. autoclass:: lbmpy.velocityspace.CustomStencil
    :members: