from __future__ import annotations

from libtbx.phil import parse

from dials.array_family import flex

phil_scope = parse(
    """
overlaps_filter {
  foreground_foreground {
    enable = False
      .type = bool
      .help = "Remove all spots in which neighbors' foreground"
              "impinges on the spot's foreground"
  }
  foreground_background {
    enable = False
      .type = bool
      .help = "Remove all spots in which neighbors' foreground"
              "impinges on the spot's background"
  }
}
""",
    process_includes=True,
)


class OverlapsFilter:
    from dials.algorithms.shoebox import MaskCode

    code_fgd = MaskCode.Foreground | MaskCode.Valid
    code_bgd = MaskCode.Background | MaskCode.Valid

    def is_fgd(self, code):
        return (code & self.code_fgd) == self.code_fgd

    def is_bgd(self, code):
        return (code & self.code_bgd) == self.code_bgd

    def __init__(self, refl, expt):
        self.refl = refl
        self.expt = expt
        self.masks = {}

        self.det = self.expt.detector
        self.panel_sizes = []
        self.panel_array_sizes = []

        for p in self.det:
            size_fast, size_slow = p.get_image_size()
            self.panel_sizes.append((size_fast, size_slow))
            self.panel_array_sizes.append(size_fast * size_slow)

    def create_simple_mask(self):
        self.masks["simple_mask"] = [{} for _ in self.panel_array_sizes]

        for obs in self.refl.rows():
            shoebox = obs["shoebox"]
            panel = obs["panel"]
            size_fast, size_slow = self.panel_sizes[panel]
            mask = self.masks["simple_mask"][panel]

            for z in range(shoebox.zsize()):
                z_abs = shoebox.bbox[4] + z
                for s in range(shoebox.ysize()):
                    s_abs = shoebox.bbox[2] + s
                    if s_abs < 0 or s_abs >= size_slow:
                        continue
                    for f in range(shoebox.xsize()):
                        f_abs = shoebox.bbox[0] + f
                        if f_abs < 0 or f_abs >= size_fast:
                            continue
                        posn_in_shoebox = (
                            z * shoebox.ysize() * shoebox.xsize()
                            + s * shoebox.xsize()
                            + f
                        )
                        key = (z_abs, s_abs, f_abs)
                        mask[key] = mask.get(key, 0) | shoebox.mask[posn_in_shoebox]

    def create_referenced_mask(self, test_code, mask_name):
        self.masks[mask_name] = [{} for _ in self.panel_array_sizes]

        for idx in range(len(self.refl)):
            obs = self.refl[idx]
            shoebox = obs["shoebox"]
            panel = obs["panel"]
            size_fast, size_slow = self.panel_sizes[panel]
            mask = self.masks[mask_name][panel]

            for z in range(shoebox.zsize()):
                z_abs = shoebox.bbox[4] + z
                for s in range(shoebox.ysize()):
                    s_abs = shoebox.bbox[2] + s
                    if s_abs < 0 or s_abs >= size_slow:
                        continue
                    for f in range(shoebox.xsize()):
                        f_abs = shoebox.bbox[0] + f
                        if f_abs < 0 or f_abs >= size_fast:
                            continue
                        posn_in_shoebox = (
                            z * shoebox.ysize() * shoebox.xsize()
                            + s * shoebox.xsize()
                            + f
                        )
                        if (shoebox.mask[posn_in_shoebox] & test_code) == test_code:
                            key = (z_abs, s_abs, f_abs)
                            if key not in mask:
                                mask[key] = []
                            mask[key].append(idx)

    def filter_using_simple_mask(self, mask_lambda, shoebox_lambda=lambda x: True):
        keep_refl_bool = flex.bool(len(self.refl), True)

        for idx in range(len(self.refl)):
            obs = self.refl[idx]
            shoebox = obs["shoebox"]
            panel = obs["panel"]
            size_fast, size_slow = self.panel_sizes[panel]
            mask = self.masks["simple_mask"][panel]

            for z in range(shoebox.zsize()):
                z_abs = shoebox.bbox[4] + z
                for s in range(shoebox.ysize()):
                    s_abs = shoebox.bbox[2] + s
                    if s_abs < 0 or s_abs >= size_slow:
                        continue
                    for f in range(shoebox.xsize()):
                        f_abs = shoebox.bbox[0] + f
                        if f_abs < 0 or f_abs >= size_fast:
                            continue
                        key = (z_abs, s_abs, f_abs)
                        if key not in mask:
                            continue
                        posn_in_shoebox = (
                            z * shoebox.ysize() * shoebox.xsize()
                            + s * shoebox.xsize()
                            + f
                        )
                        if mask_lambda(mask[key]) and shoebox_lambda(
                            shoebox.mask[posn_in_shoebox]
                        ):
                            keep_refl_bool[idx] = False

        return keep_refl_bool

    def filter_all_using_referenced_mask(self, mask_name):
        keep_refl_bool = flex.bool(len(self.refl), True)

        for panel_mask in self.masks[mask_name]:
            for refs in panel_mask.values():
                for ref in refs:
                    keep_refl_bool[ref] = False

        return keep_refl_bool

    def filter_overlaps_using_referenced_mask(self, mask_name):
        keep_refl_bool = flex.bool(len(self.refl), True)

        for panel_mask in self.masks[mask_name]:
            for refs in panel_mask.values():
                if len(refs) > 1:
                    for ref in refs:
                        keep_refl_bool[ref] = False

        return keep_refl_bool

    def remove_foreground_foreground_overlaps(self):
        self.create_referenced_mask(self.code_fgd, "foreground")
        self.refl = self.refl.select(
            self.filter_overlaps_using_referenced_mask("foreground")
        )

    def remove_foreground_background_overlaps(self):
        self.create_simple_mask()

        def is_overlap(code):
            return self.is_fgd(code) and self.is_bgd(code)

        self.refl = self.refl.select(
            self.filter_using_simple_mask(mask_lambda=is_overlap)
        )


class OverlapsFilterMultiExpt:
    def __init__(self, refl, expt):
        self.filters = [
            OverlapsFilter(r, e) for (r, e) in zip(refl.split_by_experiment_id(), expt)
        ]

    def remove_foreground_foreground_overlaps(self):
        for f in self.filters:
            f.remove_foreground_foreground_overlaps()

    def remove_foreground_background_overlaps(self):
        for f in self.filters:
            f.remove_foreground_background_overlaps()

    @property
    def refl(self):
        rlist = [f.refl for f in self.filters]
        r0 = flex.reflection_table()
        for r in rlist:
            r0.extend(r)
        return r0

    @property
    def expt(self):
        elist = [f.expt for f in self.filters]
        from dxtbx.model.experiment_list import ExperimentList

        e0 = ExperimentList()
        for e in elist:
            e0.append(e)
        return e0
