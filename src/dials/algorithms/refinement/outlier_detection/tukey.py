from __future__ import annotations

from dials.algorithms.refinement.outlier_detection import CentroidOutlier
from dials.array_family import flex


class Tukey(CentroidOutlier):
    """Implementation of the CentroidOutlier class using Tukey's rule of thumb.
    That is values more than iqr_multiplier times the interquartile range from
    the quartiles are designed outliers. When x=1.5, this is Tukey's rule."""

    def __init__(
        self,
        cols=None,
        min_num_obs=20,
        separate_experiments=True,
        separate_panels=True,
        separate_images=False,
        block_width=None,
        nproc=1,
        iqr_multiplier=1.5,
    ):

        if cols is None:
            cols = ["x_resid", "y_resid", "phi_resid"]
        CentroidOutlier.__init__(
            self,
            cols=cols,
            min_num_obs=min_num_obs,
            separate_experiments=separate_experiments,
            block_width=block_width,
            nproc=nproc,
            separate_panels=separate_panels,
            separate_images=separate_images,
        )

        self._iqr_multiplier = iqr_multiplier

        return

    def _detect_outliers(self, cols):

        from scitbx.math import five_number_summary

        outliers = flex.bool(len(cols[0]), False)
        for col in cols:
            min_x, q1_x, med_x, q3_x, max_x = five_number_summary(col)
            iqr_x = q3_x - q1_x
            cut_x = self._iqr_multiplier * iqr_x
            outliers.set_selected(col > q3_x + cut_x, True)
            outliers.set_selected(col < q1_x - cut_x, True)

        return outliers
