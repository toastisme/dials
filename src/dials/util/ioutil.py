from __future__ import annotations


def get_inverse_ub_matrix_from_xparm(handle):
    """Get the inverse_ub_matrix from an xparm file handle

    Params:
        handle The file handle

    Returns:
        The inverse_ub_matrix
    """
    from scitbx import matrix

    return matrix.sqr(
        handle.unit_cell_a_axis + handle.unit_cell_b_axis + handle.unit_cell_c_axis
    )


def get_space_group_type_from_xparm(handle):
    """Get the space group tyoe object from an xparm file handle

    Params:
        handle The file handle

    Returns:
        The space group type object
    """
    from cctbx import sgtbx

    return sgtbx.space_group_type(
        sgtbx.space_group(sgtbx.space_group_symbols(handle.space_group).hall())
    )
