'''
Tests for scaling utilities module.
'''
from math import sqrt, pi
import pytest
from mock import Mock
from dxtbx.model import Experiment, ExperimentList
from libtbx.test_utils import approx_equal
from cctbx.sgtbx import space_group
from dials.array_family import flex
from dials.algorithms.scaling.scaling_utilities import \
  calc_crystal_frame_vectors, calc_theta_phi, create_sph_harm_table,\
  sph_harm_table, align_rotation_axis_along_z, parse_multiple_datasets,\
  set_wilson_outliers, select_datasets_on_ids, assign_unique_identifiers

@pytest.fixture(scope='module')
def test_space_group():
  """Create a space group object."""
  return space_group("C 2y")

@pytest.fixture(scope='module')
def mock_exp():
  """Create a mock experiments object."""
  exp = Mock()
  exp.beam.get_s0.return_value = (1.0, 0.0, 0.0)
  exp.goniometer.get_rotation_axis.return_value = (0.0, 0.0, 1.0)
  return exp

@pytest.fixture(scope='module')
def test_reflection_table():
  """Return a test reflection table."""
  return generate_reflection_table()

@pytest.fixture(scope='module')
def wilson_test_reflection_table():
  """Return a test reflection table."""
  rt = flex.reflection_table()
  rt['centric_flag'] = flex.bool([True, True, False, False])
  rt['Esq'] = flex.double([50.0, 10.0, 50.0, 10.0])
  return rt

def generate_reflection_table():
  """Create a reflection table with s1 and phi."""
  rt = flex.reflection_table()
  s1_vec = (1.0/sqrt(2.0), 0.0, 1.0/sqrt(2.0))
  rt['s1'] = flex.vec3_double([s1_vec, s1_vec, s1_vec])
  rt['phi'] = flex.double([0.0, 45.0, 90.0])
  return rt

def test_calc_crystal_frame_vectors(test_reflection_table, mock_exp):
  """Test the namesake function, to check that the vectors are correctly rotated
  into the crystal frame."""
  rt, exp = test_reflection_table, mock_exp
  s0_vec = (1.0, 0.0, 0.0)
  s1_vec = (1.0/sqrt(2.0), 0.0, 1.0/sqrt(2.0))
  reflection_table = calc_crystal_frame_vectors(rt, exp)
  assert list(reflection_table['s0']) == list(flex.vec3_double(
    [s0_vec, s0_vec, s0_vec]))
  assert approx_equal(list(reflection_table['s0c']), list(flex.vec3_double([
    s0_vec, (1.0/sqrt(2.0), -1.0/sqrt(2.0), 0.0), (0.0, -1.0, 0.0)])))
  assert approx_equal(list(reflection_table['s1c']), list(flex.vec3_double([
    s1_vec, (1.0/2.0, -1.0/2.0, 1.0/sqrt(2.0)),
    (0.0, -1.0/sqrt(2.0), 1.0/sqrt(2.0))])))

def test_align_rotation_axis_along_z():
  """Test the function to rotate the coordinate system such that the rotation
  axis is along z. In the test, the rotation axis is x, so we expect the
  transformation to be: x > z, y > y, z > -x, x+z > -x+z."""
  rot_axis = flex.vec3_double([(1.0, 0.0, 0.0)])
  vectors = flex.vec3_double([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0)])
  rotated_vectors = align_rotation_axis_along_z(rot_axis, vectors)
  assert approx_equal(list(rotated_vectors), list(flex.vec3_double([
    (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 1.0)])))

def test_sph_harm_table(test_reflection_table, mock_exp):
  """Simple test for the spherical harmonic table, constructing the table step
  by step, and verifying the values of a few easy-to-calculate entries.
  This also acts as a test for the calc_theta_phi function as well."""
  from scitbx import sparse # Needed to be able to assign to sph_h_t
  rt, exp = test_reflection_table, mock_exp
  reflection_table = calc_crystal_frame_vectors(rt, exp)
  theta_phi = calc_theta_phi(reflection_table['s0c'])
  assert approx_equal(list(theta_phi),
    [(pi/2.0, 0.0), (pi/2.0, -1.0*pi/4.0), (pi/2.0, -1.0*pi/2.0)])
  theta_phi_2 = calc_theta_phi(reflection_table['s1c'])
  assert approx_equal(list(theta_phi_2),
    [(pi/4.0, 0.0), (pi/4.0, -1.0*pi/4.0), (pi/4.0, -1.0*pi/2.0)])
  sph_h_t = create_sph_harm_table(theta_phi, theta_phi_2, 2)
  Y10 = ((3.0/(8.0*pi))**0.5)/2.0
  Y20 = -1.0*((5.0/(256.0*pi))**0.5)
  assert approx_equal(sph_h_t[0, 1], Y10)
  assert approx_equal(sph_h_t[1, 1], Y10)
  assert approx_equal(sph_h_t[2, 1], Y10)
  assert approx_equal(sph_h_t[0, 5], Y20)
  assert approx_equal(sph_h_t[1, 5], Y20)
  assert approx_equal(sph_h_t[2, 5], Y20)
  # Now test that you get the same by just calling the function.
  sht = sph_harm_table(rt, exp, 2)
  assert approx_equal(sht[0, 1], Y10)
  assert approx_equal(sht[1, 1], Y10)
  assert approx_equal(sht[2, 1], Y10)
  assert approx_equal(sht[0, 5], Y20)
  assert approx_equal(sht[1, 5], Y20)
  assert approx_equal(sht[2, 5], Y20)

def test_parse_multiple_datasets():
  """Test the namesake function. This expects a list of reflection tables, and
  selects on the column named 'id'."""
  rt1 = flex.reflection_table()
  rt1['id'] = flex.int([0, 0, 0])
  rt2 = flex.reflection_table()
  rt2['id'] = flex.int([2, 2, 4, 4])
  single_tables, ids = parse_multiple_datasets([rt2])
  assert list(ids) == [2, 4]
  assert len(single_tables) == 2
  single_tables, ids = parse_multiple_datasets([rt1, rt2])
  assert list(ids) == [0, 2, 4]
  assert len(single_tables) == 3
  single_tables, ids = parse_multiple_datasets([rt1])
  assert len(single_tables) == 1
  assert list(ids) == [0]

  # if a duplicate id is given, then this should be detected and new ids
  # determined for all datasets.
  rt3 = flex.reflection_table()
  rt3['id'] = flex.int([2, 2])
  single_tables, ids = parse_multiple_datasets([rt1, rt2, rt3])
  assert len(single_tables) == 4
  assert list(ids) == [0, 1, 2, 3]


def empty_explist_3exp():
  experiments = ExperimentList()
  experiments.append(Experiment())
  experiments.append(Experiment())
  experiments.append(Experiment())
  return experiments

def new_reflections_3tables():
  rt1 = flex.reflection_table()
  rt1['id'] = flex.int([0, 0, 0])
  rt2 = flex.reflection_table()
  rt2['id'] = flex.int([1, 1])
  rt3 = flex.reflection_table()
  rt3['id'] = flex.int([4, 4])
  reflections = [rt1, rt2, rt3]
  return reflections

def test_assign_unique_identifiers():
  """Test the assignment of unique identifiers"""
  # Test case where none are set but refl table ids are - use refl ids
  experiments = empty_explist_3exp()
  reflections = new_reflections_3tables()
  assert list(experiments.identifiers()) == ['', '', '']
  exp, rts = assign_unique_identifiers(experiments, reflections)
  expected_identifiers = ['0', '1', '2']
  # Check that identifiers are set in experiments and reflection table.
  assert (list(exp.identifiers())) == expected_identifiers
  for i, refl in enumerate(rts):
    assert refl.experiment_identifiers()[i] == expected_identifiers[i]
    assert list(set(refl['id'])) == [i]

  # Test case where none are set but refl table ids have duplicates
  experiments = empty_explist_3exp()
  reflections = new_reflections_3tables()
  reflections[2]['id'] = flex.int([0, 0])
  exp, rts = assign_unique_identifiers(experiments, reflections)
  expected_identifiers = ['0', '1', '2']
  # Check that identifiers are set in experiments and reflection table.
  assert (list(exp.identifiers())) == expected_identifiers
  for i, refl in enumerate(rts):
    assert refl.experiment_identifiers()[i] == expected_identifiers[i]
    assert list(set(refl['id'])) == [i]

  # Test case where identifiers are already set.
  experiments = empty_explist_3exp()
  experiments[0].identifier = '0'
  experiments[1].identifier = '4'
  experiments[2].identifier = '2'
  reflections = new_reflections_3tables()
  reflections[1].experiment_identifiers()[0] = '5'
  # should raise an assertion error for inconsistent identifiers
  with pytest.raises(AssertionError):
    exp, rts = assign_unique_identifiers(experiments, reflections)

  experiments = empty_explist_3exp()
  experiments[0].identifier = '0'
  experiments[1].identifier = '4'
  experiments[2].identifier = '2'
  reflections = new_reflections_3tables()
  reflections[0].experiment_identifiers()[0] = '0'
  reflections[1].experiment_identifiers()[1] = '4'
  reflections[2].experiment_identifiers()[4] = '2'
  #should pass experiments back if identifiers all already set
  exp, rts = assign_unique_identifiers(experiments, reflections)
  expected_identifiers = ['0', '4', '2']
  # Check that identifiers are set in experiments and reflection table.
  assert exp is experiments
  assert list(exp.identifiers()) == expected_identifiers
  for i, refl in enumerate(rts):
    id_ = refl['id'][0]
    assert refl.experiment_identifiers()[id_] == expected_identifiers[i]
    assert list(set(refl['id'])) == [i]

  # Now test that if some are set, these are maintained and unique ids are
  # set for the rest
  experiments = empty_explist_3exp()
  experiments[0].identifier = '1'
  reflections = new_reflections_3tables()
  reflections[0].experiment_identifiers()[0] = '1'
  exp, rts = assign_unique_identifiers(experiments, reflections)
  expected_identifiers = ['1', '0', '2']
  assert list(exp.identifiers()) == expected_identifiers
  for i, refl in enumerate(rts):
    id_ = refl['id'][0]
    assert refl.experiment_identifiers()[id_] == expected_identifiers[i]
    assert list(set(refl['id'])) == [i]

def test_select_datasets_on_ids():
  """Test the select_datasets_on_ids function."""
  experiments = empty_explist_3exp()
  reflections = new_reflections_3tables()
  reflections[0].experiment_identifiers()[0] = '0'
  experiments[0].identifier = '0'
  reflections[1].experiment_identifiers()[1] = '2'
  experiments[1].identifier = '2'
  reflections[2].experiment_identifiers()[4] = '4'
  experiments[2].identifier = '4'
  use_datasets = ['0', '2']
  exp, refl = select_datasets_on_ids(experiments, reflections,
    use_datasets=use_datasets)
  assert len(exp) == 2
  assert len(refl) == 2
  assert list(exp.identifiers()) == ['0', '2']

  exclude_datasets = ['0']
  exp, refl = select_datasets_on_ids(experiments, reflections,
    exclude_datasets=exclude_datasets)
  assert len(refl) == 2
  assert list(exp.identifiers()) == ['2', '4']
  assert len(exp) == 2

  with pytest.raises(AssertionError):
    exclude_datasets = ['0']
    use_datasets = ['2', '4']
    exp, refl = select_datasets_on_ids(experiments,
      reflections, use_datasets=use_datasets,
      exclude_datasets=exclude_datasets)

  with pytest.raises(AssertionError):
    exclude_datasets = ['1']
    exp, refl = select_datasets_on_ids(experiments,
      reflections, exclude_datasets=exclude_datasets)


def set_calculate_wilson_outliers(wilson_test_reflection_table):
  """Test the set wilson outliers function."""
  reflection_table = set_wilson_outliers(wilson_test_reflection_table)

  assert list(reflection_table.get_flags(
    reflection_table.flags.outlier_in_scaling)) == [True, False, True, False]