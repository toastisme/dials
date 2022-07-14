from __future__ import annotations

from math import pi

from dxtbx.model import Crystal
from rstbx.symmetry.constraints.parameter_reduction import symmetrize_reduce_enlarge
from scitbx import matrix

from dials.algorithms.refinement.parameterisation.crystal_parameters import (
    CrystalOrientationParameterisation,
    CrystalUnitCellParameterisation,
)
from dials.algorithms.refinement.refinement_helpers import (
    get_fd_gradients,
    random_param_shift,
)


def test():
    import random
    import textwrap

    from cctbx.uctbx import unit_cell
    from libtbx.test_utils import approx_equal

    def random_direction_close_to(vector):
        return vector.rotate_around_origin(
            matrix.col((random.random(), random.random(), random.random())).normalize(),
            random.gauss(0, 1.0),
            deg=True,
        )

    # make a random P1 crystal and parameterise it
    a = random.uniform(10, 50) * random_direction_close_to(matrix.col((1, 0, 0)))
    b = random.uniform(10, 50) * random_direction_close_to(matrix.col((0, 1, 0)))
    c = random.uniform(10, 50) * random_direction_close_to(matrix.col((0, 0, 1)))
    xl = Crystal(a, b, c, space_group_symbol="P 1")

    xl_op = CrystalOrientationParameterisation(xl)
    xl_ucp = CrystalUnitCellParameterisation(xl)

    null_mat = matrix.sqr((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # compare analytical and finite difference derivatives
    an_ds_dp = xl_op.get_ds_dp()
    fd_ds_dp = get_fd_gradients(xl_op, [1.0e-6 * pi / 180] * 3)
    for e, f in zip(an_ds_dp, fd_ds_dp):
        assert approx_equal((e - f), null_mat, eps=1.0e-6)

    an_ds_dp = xl_ucp.get_ds_dp()
    fd_ds_dp = get_fd_gradients(xl_ucp, [1.0e-7] * xl_ucp.num_free())
    for e, f in zip(an_ds_dp, fd_ds_dp):
        assert approx_equal((e - f), null_mat, eps=1.0e-6)

    # random initial orientations with a random parameter shift at each
    attempts = 100
    for i in range(attempts):

        # make a random P1 crystal and parameterise it
        a = random.uniform(10, 50) * random_direction_close_to(matrix.col((1, 0, 0)))
        b = random.uniform(10, 50) * random_direction_close_to(matrix.col((0, 1, 0)))
        c = random.uniform(10, 50) * random_direction_close_to(matrix.col((0, 0, 1)))
        xl = Crystal(a, b, c, space_group_symbol="P 1")
        xl_op = CrystalOrientationParameterisation(xl)
        xl_uc = CrystalUnitCellParameterisation(xl)

        # apply a random parameter shift to the orientation
        p_vals = xl_op.get_param_vals()
        p_vals = random_param_shift(
            p_vals, [1000 * pi / 9, 1000 * pi / 9, 1000 * pi / 9]
        )
        xl_op.set_param_vals(p_vals)

        # compare analytical and finite difference derivatives
        xl_op_an_ds_dp = xl_op.get_ds_dp()
        xl_op_fd_ds_dp = get_fd_gradients(xl_op, [1.0e-5 * pi / 180] * 3)

        # apply a random parameter shift to the unit cell. We have to
        # do this in a way that is respectful to metrical constraints,
        # so don't modify the parameters directly; modify the cell
        # constants and extract the new parameters
        cell_params = xl.get_unit_cell().parameters()
        cell_params = random_param_shift(cell_params, [1.0] * 6)
        new_uc = unit_cell(cell_params)
        newB = matrix.sqr(new_uc.fractionalization_matrix()).transpose()
        S = symmetrize_reduce_enlarge(xl.get_space_group())
        S.set_orientation(orientation=newB)
        X = S.forward_independent_parameters()
        xl_uc.set_param_vals(X)

        xl_uc_an_ds_dp = xl_ucp.get_ds_dp()

        # now doing finite differences about each parameter in turn
        xl_uc_fd_ds_dp = get_fd_gradients(xl_ucp, [1.0e-7] * xl_ucp.num_free())

        for j in range(3):
            assert approx_equal(
                (xl_op_fd_ds_dp[j] - xl_op_an_ds_dp[j]), null_mat, eps=1.0e-6
            ), textwrap.dedent(
                """\
        Failure in try {i}
        failure for parameter number {j}
        of the orientation parameterisation
        with fd_ds_dp =
        {fd}
        and an_ds_dp =
        {an}
        so that difference fd_ds_dp - an_ds_dp =
        {diff}
        """
            ).format(
                i=i,
                j=j,
                fd=xl_op_fd_ds_dp[j],
                an=xl_op_an_ds_dp[j],
                diff=xl_op_fd_ds_dp[j] - xl_op_an_ds_dp[j],
            )

        for j in range(xl_ucp.num_free()):
            assert approx_equal(
                (xl_uc_fd_ds_dp[j] - xl_uc_an_ds_dp[j]), null_mat, eps=1.0e-6
            ), textwrap.dedent(
                """\
        Failure in try {i}
        failure for parameter number {j}
        of the unit cell parameterisation
        with fd_ds_dp =
        {fd}
        and an_ds_dp =
        {an}
        so that difference fd_ds_dp - an_ds_dp =
        {diff}
        """
            ).format(
                i=i,
                j=j,
                fd=xl_uc_fd_ds_dp[j],
                an=xl_uc_an_ds_dp[j],
                diff=xl_uc_fd_ds_dp[j] - xl_uc_an_ds_dp[j],
            )
