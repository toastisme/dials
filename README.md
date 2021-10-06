# Extending DIALS to Neutron Diffraction Data

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/dials/dials.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/dials/dials/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/dials/dials.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/dials/dials/alerts/)
[![Coverage](https://codecov.io/gh/dials/dials/branch/main/graph/badge.svg)](https://codecov.io/gh/dials/dials)
[![Gitter](https://badges.gitter.im/dials/community.svg)](https://gitter.im/dials/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

The DIALS project provides an extensible framework to analyse X-ray diffraction data. Much of this work is agnostic to the method used to obtain diffraction patterns, and can equally be applied to neutron diffraction data. This fork is a work in progress for extending DIALS to neutron diffraction experiments. Initial work is focused on processing time-of-flight (ToF) Laue data from SXD at ISIS. This is expected to be in production by the second quarter of 2022, with plans to extend the work to monochromatic and quasi-Laue neutron experiments in the near future.

Required Repos
-------
[dxtbx fork](https://github.com/toastisme/dxtbx) | [cctbx fork](https://github.com/toastisme/cctbx_project)

Developer Installation
-------
Follow the [DIALS installation for developers](https://dials.github.io/documentation/installation_developer.html), clone the [DIALS](https://github.com/toastisme/dials), [dxtbx](https://github.com/toastisme/dxtbx), and [cctbx](https://github.com/toastisme/cctbx_project) forks, and replace the corresponding repos in `dials/modules` with these. Then run `python dials/bootstrap.py build`.

Example data is available on [zenodo](https://doi.org/10.5281/zenodo.4415768). For ToF Laue neutron data DIALS is designed to work with the [NeXus TOFRAW format](https://www.nexusformat.org/TOFRaw.html). [ISIS Raw files](https://www.isis.stfc.ac.uk/Pages/ISIS-Raw-File-Format.aspx) can be converted to TOFRAW using [isis_utils](https://github.com/dials/isis_utils).

DIALS Website
-------

https://dials.github.io


Reference
---------

[Winter, G., Waterman, D. G., Parkhurst, J. M., Brewster, A. S., Gildea, R. J., Gerstel, M., Fuentes-Montero, L., Vollmar, M., Michels-Clark, T., Young, I. D., Sauter, N. K. and Evans, G. (2018) Acta Cryst. D74.](http://journals.iucr.org/d/issues/2018/02/00/di5011/index.html)

Funding
-------

DIALS development at Diamond Light Source is supported by the BioStruct-X EU grant, Diamond Light Source, and CCP4.

DIALS development at Lawrence Berkeley National Laboratory is supported by National Institutes of Health / National Institute of General Medical Sciences grant R01-GM117126. Work at LBNL is performed under Department of Energy contract DE-AC02-05CH11231.

The foundational work in extending DIALS to neutron diffraction for SXD data is supported by the Ada Lovelace Centre.
