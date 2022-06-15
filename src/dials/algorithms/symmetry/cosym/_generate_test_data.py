from __future__ import annotations

import random

import numpy as np

import scitbx.matrix
import scitbx.random
from cctbx import crystal, sgtbx
from cctbx.sgtbx.subgroups import subgroups
from dxtbx.model import Crystal, Experiment, ExperimentList, Scan

from dials.array_family import flex


def generate_experiments_reflections(
    space_group,
    lattice_group=None,
    unit_cell=None,
    unit_cell_volume=1000,
    seed=0,
    d_min=1,
    sigma=0.1,
    sample_size=100,
    map_to_p1=False,
    twin_fractions=None,
    map_to_minimum=True,
):
    datasets, reindexing_ops = generate_test_data(
        space_group,
        lattice_group=lattice_group,
        unit_cell=unit_cell,
        unit_cell_volume=unit_cell_volume,
        seed=seed,
        d_min=d_min,
        sigma=sigma,
        sample_size=sample_size,
        map_to_p1=map_to_p1,
        twin_fractions=twin_fractions,
        map_to_minimum=map_to_minimum,
    )

    expts = ExperimentList()
    refl_tables = []

    for i, dataset in enumerate(datasets):
        B = scitbx.matrix.sqr(
            dataset.unit_cell().fractionalization_matrix()
        ).transpose()
        expts.append(
            Experiment(
                crystal=Crystal(B, space_group=dataset.space_group(), reciprocal=True),
                sequence=Scan(image_range=(0, 180), oscillation=(0.0, 1.0)),
            )
        )
        refl = flex.reflection_table()
        refl["miller_index"] = dataset.indices()
        refl["id"] = flex.int(refl.size(), i)
        refl["d"] = dataset.d_spacings().data()
        refl["intensity.sum.value"] = dataset.data()
        refl["intensity.sum.variance"] = flex.pow2(dataset.sigmas())
        refl.set_flags(flex.bool(len(refl), True), refl.flags.integrated_sum)
        refl_tables.append(refl)
    return expts, refl_tables, reindexing_ops


def generate_test_data(
    space_group,
    lattice_group=None,
    unit_cell=None,
    unit_cell_volume=1000,
    seed=0,
    d_min=1,
    sigma=0.1,
    sample_size=100,
    map_to_p1=False,
    twin_fractions=None,
    map_to_minimum=True,
):
    if seed is not None:
        flex.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    assert [unit_cell, lattice_group].count(None) > 0

    sgi = space_group.info()

    if unit_cell is not None:
        cs = crystal.symmetry(unit_cell=unit_cell, space_group_info=sgi)
    elif lattice_group is not None:
        subgrps = subgroups(lattice_group).groups_parent_setting()
        assert space_group in subgrps
        cs = lattice_group.any_compatible_crystal_symmetry(
            volume=unit_cell_volume
        ).customized_copy(space_group_info=sgi)
    else:
        cs = sgi.any_compatible_crystal_symmetry(volume=unit_cell_volume)

    if map_to_minimum:
        cs = cs.minimum_cell()
    intensities = generate_intensities(cs, d_min=d_min)
    intensities.show_summary()

    twin_ops = generate_twin_operators(intensities)
    twin_ops = [sgtbx.change_of_basis_op(op.operator.as_xyz()) for op in twin_ops]

    if twin_fractions is not None:
        assert len(twin_fractions) == len(twin_ops)
        assert len(twin_fractions) == 1, "Only 1 twin component currently supported"
        twin_op = twin_ops[0]
        twin_fraction = twin_fractions[0]
        intensities, intensities_twin = intensities.common_sets(
            intensities.change_basis(twin_op).map_to_asu()
        )
        twinned_miller = intensities.customized_copy(
            data=(1.0 - twin_fraction) * intensities.data()
            + twin_fraction * intensities_twin.data(),
            sigmas=flex.sqrt(
                flex.pow2((1.0 - twin_fraction) * intensities.sigmas())
                + flex.pow2(twin_fraction * intensities_twin.sigmas())
            ),
        )
        intensities = twinned_miller

    cb_ops = twin_ops
    cb_ops.insert(0, sgtbx.change_of_basis_op())

    reindexing_ops = []

    datasets = []
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=sigma)
    g = scitbx.random.variate(rand_norm)
    for i in range(sample_size):
        cb_op = random.choice(cb_ops)
        reindexing_ops.append(cb_op.as_xyz())
        d = intensities.change_basis(cb_op).customized_copy(
            crystal_symmetry=intensities.crystal_symmetry()
        )

        if map_to_p1:
            cb_op_to_primitive = d.change_of_basis_op_to_primitive_setting()
            d = d.change_basis(cb_op_to_primitive)
            d = d.expand_to_p1()

        d = d.customized_copy(data=d.data() + g(d.size()))
        datasets.append(d)

    return datasets, reindexing_ops


def generate_intensities(crystal_symmetry, anomalous_flag=False, d_min=1):
    from cctbx import miller

    indices = miller.index_generator(
        crystal_symmetry.unit_cell(),
        crystal_symmetry.space_group().type(),
        anomalous_flag,
        d_min,
    ).to_array()
    miller_set = crystal_symmetry.miller_set(indices, anomalous_flag)
    intensities = flex.random_double(indices.size()) * 1000
    miller_array = miller.array(
        miller_set, data=intensities, sigmas=flex.sqrt(intensities)
    ).set_observation_type_xray_intensity()
    return miller_array


def generate_twin_operators(miller_array, verbose=True):
    from mmtbx.scaling.twin_analyses import twin_laws

    TL = twin_laws(miller_array=miller_array)
    return TL.operators
