import logging

logger = logging.getLogger(__name__)


class MaskerBase:
    """A root class to that does overlap masking"""

    def __init__(self, experiment):
        """Initialise the overlap masking algorithm

        Params:
            experiment The experiment data
        """
        from dials.algorithms.shoebox import MaskOverlapping

        # Construct the overlapping reflection mask
        self.mask_overlapping = MaskOverlapping()

    def __call__(self, reflections, adjacency_list=None):
        """Mask the given reflections.

        Params:
            reflections The reflection list
            adjacency_list The adjacency_list (optional)

        Returns:
            The masked reflection list
        """

        # Mask the overlaps if an adjacency list is given
        if adjacency_list:
            logger.info("Masking overlapping reflections")
            self.mask_overlapping(
                reflections["shoebox"], reflections["xyzcal.px"], adjacency_list
            )
            logger.info("Masked %s overlapping reflections", len(adjacency_list))

        # Return the reflections
        return reflections


class Masker3DProfile(MaskerBase):
    """A class to perform 3D profile masking"""

    def __init__(self, experiments):
        """Initialise the masking algorithms

        Params:
            experiment The experiment data
            delta_d The extent of the reflection in reciprocal space
            delta_m The extent of the reflection in reciprocal space
        """
        super().__init__(experiments[0])
        self._experiments = experiments

    def __call__(self, reflections, adjacency_list=None):
        """Mask the given reflections.

        Params:
            reflections The reflection list
            adjacency_list The adjacency_list (optional)

        Returns:
            The masked reflection list
        """
        reflections = super().__call__(reflections, adjacency_list)

        # Mask the foreground region
        self._experiment.profile.compute_mask(
            reflections,
            self._experiment.crystal,
            self._experiment.beam,
            self._experiment.detector,
            self._experiment.goniometer,
            self._experiment.sequence,
        )

        # Return the reflections
        return reflections


class MaskerEmpirical(MaskerBase):
    """A class to perform empirical masking"""

    def __init__(self, experiment, reference):
        """Initialise the masking algorithms

        Params:
            experiment The experiment data
        """
        super().__init__(experiment)

        from dials.algorithms.shoebox import MaskEmpirical

        # Construct the foreground pixel mask
        self.mask_empirical = MaskEmpirical(reference)
        self._reference = reference

    def __call__(self, reflections, adjacency_list=None):
        """Mask the given reflections.

        Params:
            reflections The reflection list
            adjacency_list The adjacency_list (optional)

        Returns:
            The masked reflection list
        """
        reflections = super().__call__(reflections, adjacency_list)

        if self.mask_empirical:
            # Mask the foreground region
            self.mask_empirical(reflections)

        # Return the reflections
        return reflections
