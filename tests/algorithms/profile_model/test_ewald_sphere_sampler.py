def test_run(dials_data):
    experiments = dials_data("centroid_test_data").join("experiments.json")

    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(experiments.strpath)

    from dials.algorithms.profile_model.modeller import EwaldSphereSampler

    beam = experiments[0].beam
    detector = experiments[0].detector
    goniometer = experiments[0].goniometer
    scan = experiments[0].sequence

    sampler = EwaldSphereSampler(beam, detector, goniometer, scan, 1)

    assert sorted(sampler.nearest_n(0)) == sorted([0, 1, 2, 3, 4, 5, 6, 7, 8])

    assert sorted(sampler.nearest_n(1)) == sorted([0, 1, 2, 8, 9, 10])
    assert sorted(sampler.nearest_n(2)) == sorted([0, 2, 3, 1, 11, 12])
    assert sorted(sampler.nearest_n(3)) == sorted([0, 3, 4, 2, 13, 14])
    assert sorted(sampler.nearest_n(4)) == sorted([0, 4, 5, 3, 15, 16])
    assert sorted(sampler.nearest_n(5)) == sorted([0, 5, 6, 4, 17, 18])
    assert sorted(sampler.nearest_n(6)) == sorted([0, 6, 7, 5, 19, 20])
    assert sorted(sampler.nearest_n(7)) == sorted([0, 7, 8, 6, 21, 22])
    assert sorted(sampler.nearest_n(8)) == sorted([0, 8, 1, 7, 23, 24])

    assert sorted(sampler.nearest_n(9)) == sorted([1, 9, 10, 24, 25, 26])
    assert sorted(sampler.nearest_n(10)) == sorted([1, 10, 11, 9, 27, 28])
    assert sorted(sampler.nearest_n(24)) == sorted([8, 24, 9, 23, 55, 56])

    assert sorted(sampler.nearest_n(25)) == sorted([9, 25, 26, 56])
    assert sorted(sampler.nearest_n(26)) == sorted([9, 26, 27, 25])
    assert sorted(sampler.nearest_n(56)) == sorted([24, 56, 25, 55])

    # from scitbx import matrix
    # from math import cos, sin
    # s0 = matrix.col(beam.get_s0()).normalize()
    # m2 = matrix.col(goniometer.get_rotation_axis()).normalize()
    # zaxis = s0
    # yaxis = zaxis.cross(m2)
    # xaxis = zaxis.cross(yaxis)
    # points_x = []
    # points_y = []
    # for i in range(len(sampler)):
    #   a,b,phi = sampler.profile_coord(i)
    #   x = sin(a)*cos(b)
    #   y = sin(a)*sin(b)
    #   z = cos(a)
    #   s1 = x * xaxis + y*yaxis + z*zaxis
    #   try:
    #     px, py = detector[0].get_ray_intersection_px(s1)
    #     points_x.append(px)
    #     points_y.append(py)
    #   except Exception:
    #     pass

    # width, height = detector[0].get_image_size()
    # from dials.array_family import flex
    # image = flex.double(flex.grid(height, width))

    # for j in range(height):
    #   for i in range(width):
    #     coord = (i, j, 0.5)

    #     index = sampler.nearest(coord)
    #     image[j,i] = sampler.weight(index, coord)

    # from matplotlib import pylab
    # pylab.imshow(image.as_numpy_array())
    # pylab.colorbar()
    # pylab.scatter(points_x, points_y)
    # pylab.show()
