from __future__ import annotations


class ProfileModelExt:
    """
    The definition for a profile model.
    """

    @staticmethod
    def create(
        params, reflections, crystal, beam, detector, goniometer=None, scan=None
    ):
        """
        Create the profile model from data.

        :param params: The phil parameters
        :param reflections: The reflections
        :param crystal: The crystal model
        :param beam: The beam model
        :param detector: The detector model
        :param goniometer: The goniometer model
        :param scan: The scan model
        :return: An instance of the profile model
        """
        return None

    def predict_reflections(
        self, imageset, crystal, beam, detector, goniometer=None, scan=None, **kwargs
    ):
        """
        Given an experiment, predict the reflections.

        :param imageset: The imageset
        :param crystal: The crystal model
        :param beam: The beam model
        :param detector: The detector model
        :param goniometer: The goniometer model
        :param scan: The scan model
        """
        pass

    def compute_partiality(
        self, reflections, crystal, beam, detector, goniometer=None, scan=None, **kwargs
    ):
        """
        Given an experiment and list of reflections, compute the partiality of the
        reflections

        :param reflections: The reflection table
        :param crystal: The crystal model
        :param beam: The beam model
        :param detector: The detector model
        :param goniometer: The goniometer model
        :param scan: The scan model
        """
        pass

    def compute_bbox(
        self, reflections, crystal, beam, detector, goniometer=None, scan=None, **kwargs
    ):
        """Given an experiment and list of reflections, compute the
        bounding box of the reflections on the detector (and image frames).

        :param reflections: The reflection table
        :param crystal: The crystal model
        :param beam: The beam model
        :param detector: The detector model
        :param goniometer: The goniometer model
        :param scan: The scan model
        """
        pass

    def compute_mask(
        self, reflections, crystal, beam, detector, goniometer=None, scan=None, **kwargs
    ):
        """
        Given an experiment and list of reflections, compute the
        foreground/background mask of the reflections.

        :param reflections: The reflection table
        :param crystal: The crystal model
        :param beam: The beam model
        :param detector: The detector model
        :param goniometer: The goniometer model
        :param scan: The scan model
        """
        pass

    def fitting_class(self):
        """
        Get the profile fitting algorithm associated with this profile model

        :return: The profile fitting class
        """
        return None
