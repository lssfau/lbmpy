import sympy as sp
from dataclasses import dataclass

from lbmpy.enums import Method
from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from pystencils import Assignment, AssignmentCollection
from pystencils.field import Field


@dataclass
class PSMConfig:
    fraction_field: Field = None
    """
    Fraction field for PSM 
    """

    fraction_field_symbol = sp.Symbol('B')
    """
    Fraction field symbol used for simplification 
    """

    object_velocity_field: Field = None
    """
    Object velocity field for PSM 
    """

    solid_collision: int = 1
    """
    Solid collision option for PSM
    """

    max_particles_per_cell: int = 1
    """
    Maximum number of particles overlapping with a cell 
    """

    individual_fraction_field: Field = None
    """
    Fraction field for each overlapping object / particle in PSM 
    """

    object_force_field: Field = None
    """
    Force field for each overlapping object / particle in PSM 
    """


def get_psm_solid_collision_term(collision_rule, psm_config, particle_per_cell_counter):
    if psm_config.individual_fraction_field is None:
        fraction_field = psm_config.fraction_field
    else:
        fraction_field = psm_config.individual_fraction_field

    method = collision_rule.method
    pre_collision_pdf_symbols = method.pre_collision_pdf_symbols
    stencil = method.stencil

    solid_collisions = [0] * stencil.Q
    equilibrium_fluid = method.get_equilibrium_terms()
    equilibrium_solid = []

    # get equilibrium form object velocity
    for eq in equilibrium_fluid:
        eq_sol = eq
        for i in range(stencil.D):
            eq_sol = eq_sol.subs(sp.Symbol("u_" + str(i)),
                                 psm_config.object_velocity_field.center(particle_per_cell_counter * stencil.D + i), )
        equilibrium_solid.append(eq_sol)

    # Build solid collision
    for i, (eqFluid, eqSolid, f, offset) in enumerate(
            zip(equilibrium_fluid, equilibrium_solid, pre_collision_pdf_symbols, stencil)):
        inverse_direction_index = stencil.stencil_entries.index(stencil.inverse_stencil_entries[i])
        if psm_config.solid_collision == 1:
            solid_collision = (fraction_field.center(particle_per_cell_counter)
                               * ((pre_collision_pdf_symbols[inverse_direction_index]
                                   - equilibrium_fluid[inverse_direction_index]) - (f - eqSolid)))
        elif psm_config.solid_collision == 2:
            # TODO get relaxation rate vector from method and use the right relaxation rate [i]
            solid_collision = (fraction_field.center(particle_per_cell_counter)
                               * ((eqSolid - f) + (1.0 - method.relaxation_rates[0]) * (f - eqFluid)))
        elif psm_config.solid_collision == 3:
            solid_collision = (fraction_field.center(particle_per_cell_counter)
                               * ((pre_collision_pdf_symbols[inverse_direction_index]
                                   - equilibrium_solid[inverse_direction_index]) - (f - eqSolid)))
        else:
            raise ValueError("Only sc=1, sc=2 and sc=3 are supported.")

        solid_collisions[i] += solid_collision

    return solid_collisions


def get_psm_force_from_solid_collision(solid_collisions, stencil, object_force_field, particle_per_cell_counter):
    force_assignments = []
    for d in range(stencil.D):
        forces_rhs = 0
        for sc, offset in zip(solid_collisions, stencil):
            forces_rhs -= sc * int(offset[d])

        force_assignments.append(Assignment(
            object_force_field.center(particle_per_cell_counter * stencil.D + d), forces_rhs
        ))
    return AssignmentCollection(force_assignments)


def replace_fraction_symbol_with_field(assignments, fraction_field_symbol, fraction_field_access):
    new_assignments = []
    for ass in assignments:
        rhs = ass.rhs.subs(fraction_field_symbol, fraction_field_access.center(0))
        new_assignments.append(Assignment(ass.lhs, rhs))
    return new_assignments


def add_psm_solid_collision_to_collision_rule(collision_rule, lbm_config, particle_per_cell_counter):

    method = collision_rule.method
    solid_collisions = get_psm_solid_collision_term(collision_rule, lbm_config.psm_config, particle_per_cell_counter)
    post_collision_pdf_symbols = method.post_collision_pdf_symbols

    assignments = []
    for sc, post in zip(solid_collisions, post_collision_pdf_symbols):
        assignments.append(Assignment(post, post + sc))

    if lbm_config.psm_config.object_force_field is not None:
        assignments += get_psm_force_from_solid_collision(solid_collisions, method.stencil,
                                                          lbm_config.psm_config.object_force_field,
                                                          particle_per_cell_counter)

    # exchanging rho with zeroth order moment symbol
    if lbm_config.method in (Method.CENTRAL_MOMENT, Method.MONOMIAL_CUMULANT, Method.CUMULANT):
        new_assignments = []
        zeroth_moment_symbol = 'm_00' if lbm_config.stencil.D == 2 else 'm_000'
        for ass in assignments:
            new_assignments.append(ass.subs(sp.Symbol('rho'), sp.Symbol(zeroth_moment_symbol)))
        assignments = new_assignments

    collision_assignments = AssignmentCollection(assignments)
    ac = LbmCollisionRule(method, collision_assignments, [],
                          collision_rule.simplification_hints)
    return ac


def replace_by_psm_collision_rule(collision_rule, psm_config):

    method = collision_rule.method
    collision_assignments = []
    solid_collisions = [0] * psm_config.max_particles_per_cell
    for p in range(psm_config.max_particles_per_cell):
        solid_collisions[p] = get_psm_solid_collision_term(collision_rule, psm_config, p)

        if psm_config.object_force_field is not None:
            collision_assignments += get_psm_force_from_solid_collision(solid_collisions[p], method.stencil,
                                                                        psm_config.object_force_field, p)

    for i, main in enumerate(collision_rule.main_assignments):
        rhs = main.rhs
        for p in range(psm_config.max_particles_per_cell):
            rhs += solid_collisions[p][i]
        collision_assignments.append(Assignment(main.lhs, rhs))

    collision_assignments = AssignmentCollection(collision_assignments)
    ac = LbmCollisionRule(method, replace_fraction_symbol_with_field(collision_assignments,
                          psm_config.fraction_field_symbol, psm_config.fraction_field),
                          replace_fraction_symbol_with_field(collision_rule.subexpressions,
                          psm_config.fraction_field_symbol, psm_config.fraction_field),
                          collision_rule.simplification_hints)
    ac.topological_sort()
    return ac
