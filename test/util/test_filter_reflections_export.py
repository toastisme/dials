"""
Test extracting of reflection intensities for export.

Two main use cases supported:

- exporting after scaling, should export only the 'scale' intensities and
  variances with the inverse scale correction applied.
  dials.scale should choose the best intensities, filter outliers and apply all
  corrections except the inverse scale factor to create the
  'intensity.scale.value', so all that is needed is to apply the inverse scales
  and propagate any errors

- exporting after integration, to send for scaling outside of dials. In this
  case, one needs to decide which intensities to export, prf, sum or both.
  this can be split into two further cases:
  - export only prf or only sum
  - export both prf and sum, even if both might not be defined for all
    reflections, so that downstream programs can decide how to use both
    intensities

As well as choosing the correct intensities, further filtering may be applied
- removing ice rings
- only exporting above a minimimum I/sigma
- filter out 'bad' values such as negative variances, negative scale factors

Lastly, partials should be correctly handled - being combined if appropriate,
scaled and filtered below a given value where the values become unreliable.
"""

import pytest
import mock
from libtbx.utils import Sorry
from dials.array_family import flex
from dials.util.filter_reflections import \
  FilteringReductionMethods, FilterForExportAlgorithm, SumIntensityReducer, \
  PrfIntensityReducer, SumAndPrfIntensityReducer, ScaleIntensityReducer, \
  filter_reflection_table, sum_partial_reflections, _sum_prf_partials, \
  _sum_sum_partials, _sum_scale_partials, AllSumPrfScaleIntensityReducer

def generate_simple_table():
  """Generate a simple table for testing export function."""
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0, 3.0])
  r['intensity.prf.variance'] = flex.double([1.0, 1.0, 1.0])
  r['partiality'] = flex.double([1.0, 1.0, 0.5])
  r['partial_id'] = flex.int([0, 1, 2])
  r['id'] = flex.int(3, 0)
  r.set_flags(flex.bool([True, True, True]), r.flags.integrated_prf)
  r.set_flags(flex.bool([True, False, False]), r.flags.in_powder_ring)
  return r

def generate_integrated_test_reflections():
  """Generate an example reflection table that would be generated by
  dials.integrate or dials.scale. This is a simple example with no partials or
  corrections, to test the correct selection of intensities."""
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  r['intensity.sum.value'] = flex.double([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
  r['intensity.scale.value'] = flex.double([21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
  r['intensity.prf.variance'] = flex.double([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  r['intensity.sum.variance'] = flex.double([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
  r['intensity.scale.variance'] = flex.double([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
  r['inverse_scale_factor'] = flex.double([5.0, 5.0, 5.0, 10.0, 10.0, -10.0])
  r['inverse_scale_factor_variance'] = flex.double([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
  r.set_flags(flex.bool([True, True, True, False, False, True]),
    r.flags.integrated_prf)
  r.set_flags(flex.bool([True, True, True, True, False, False]),
    r.flags.integrated_sum)
  r.set_flags(flex.bool([True, False, False, False, False, False]),
    r.flags.excluded_for_scaling)
  r.set_flags(flex.bool([False, True, False, False, False, False]),
    r.flags.outlier_in_scaling)
  r['id'] = flex.int(6, 0)
  r['partial_id'] = flex.int([0, 1, 2, 3, 4, 5])
  return r

def generate_test_reflections_for_scaling():
  """Generate a test table for scaling corrections."""
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
  r['intensity.sum.value'] = flex.double([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
  r['intensity.scale.value'] = flex.double([21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
  r['intensity.prf.variance'] = flex.double([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
  r['intensity.sum.variance'] = flex.double([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
  r['intensity.scale.variance'] = flex.double([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])
  r['inverse_scale_factor'] = flex.double([5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0])
  r['inverse_scale_factor_variance'] = flex.double([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
  r['lp'] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])
  r['qe'] = flex.double([1.0, 1.0, 1.0, 1.0, 0.25, 1.0, 0.0])
  r['partiality'] = flex.double([0.1, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  r['id'] = flex.int(7, 0)
  r['partial_id'] = flex.int([0, 1, 2, 3, 4, 5, 6])
  return r

fpath = 'dials.util.filter_reflections'

def test_IntensityReducer_instantiations():
  """Test that all classes can be instantiated (have the required implemented
  methods) and have an intensities list with at least one str values"""
  allowed_intensities = FilterForExportAlgorithm.allowed_intensities
  SumIntensityReducer()
  assert all([i in allowed_intensities for i in SumIntensityReducer.intensities])
  PrfIntensityReducer()
  assert all([i in allowed_intensities for i in PrfIntensityReducer.intensities])
  SumAndPrfIntensityReducer()
  assert all([i in allowed_intensities for i in SumAndPrfIntensityReducer.intensities])
  ScaleIntensityReducer()
  assert all([i in allowed_intensities for i in ScaleIntensityReducer.intensities])

def test_FilteringReductionMethods():
  """Test the FilteringReductionMethods class."""
  #Test ice ring filtering
  reflections = generate_simple_table()
  reflections = FilteringReductionMethods.filter_ice_rings(reflections)
  assert list(reflections['intensity.prf.value']) == [2.0, 3.0]

  # Test filtering on I/sigI
  reflections = generate_simple_table()
  reflections = FilteringReductionMethods._filter_on_min_isigi(
    reflections, 'prf', 2.5)
  assert list(reflections['intensity.prf.value']) == [3.0]

  # Test bad variance filtering
  reflections = generate_simple_table()
  reflections['intensity.prf.variance'][0] = 0.0
  reflections = FilteringReductionMethods._filter_bad_variances(
    reflections, 'prf')
  assert list(reflections['intensity.prf.value']) == [2.0, 3.0]

  # test calculate_lp_qe_correction_and_filter - should be lp/qe
  # cases, qe, dqe , lp , qe zero
  r = flex.reflection_table()
  r['data'] = flex.double([1.0, 2.0, 3.0])
  r, c = FilteringReductionMethods.calculate_lp_qe_correction_and_filter(r)
  assert list(c) == pytest.approx([1.0, 1.0, 1.0])

  r['lp'] = flex.double([1.0, 0.5, 1.0])
  r, c = FilteringReductionMethods.calculate_lp_qe_correction_and_filter(r)
  assert list(c) == pytest.approx([1.0, 0.5, 1.0])

  r['qe'] = flex.double([0.25, 1.0, 0.0])
  r, c = FilteringReductionMethods.calculate_lp_qe_correction_and_filter(r)
  assert list(c) == pytest.approx([4.0, 0.5])
  del r['qe']
  r['dqe'] = flex.double([0.25, 0.0])
  r, c = FilteringReductionMethods.calculate_lp_qe_correction_and_filter(r)
  assert list(c) == pytest.approx([4.0])

  # test filter unassigned
  r = flex.reflection_table()
  r['id'] = flex.int([-1, 0])
  r['i'] = flex.double([1.0, 2.0])
  r = FilteringReductionMethods.filter_unassigned_reflections(r)
  assert list(r['i']) == [2.0]

  with mock.patch(fpath+'.sum_partial_reflections',
    side_effect=return_reflections_side_effect) as sum_partials:
    reflections = generate_simple_table()
    reflections = FilteringReductionMethods.combine_and_filter_partials(
      reflections, partiality_threshold=0.7)
    assert sum_partials.call_count == 1
    assert list(reflections['intensity.prf.value']) == [1.0, 2.0]
    reflections = generate_simple_table()
    FilteringReductionMethods.combine_and_filter_partials(
      reflections, partiality_threshold=0.4)
    assert sum_partials.call_count == 2
    assert list(reflections['intensity.prf.value']) == [1.0, 2.0, 3.0]

def test_PrfIntensityReducer():
  """Test the methods of the PrfIntensityReducer class"""
  reflections = generate_integrated_test_reflections()
  reflections = PrfIntensityReducer.reduce_on_intensities(reflections)
  assert list(reflections['intensity.prf.value']) == [1.0, 2.0, 3.0, 6.0]
  assert list(reflections['intensity.prf.variance']) == [0.1, 0.2, 0.3, 0.6]

  reflections = generate_test_reflections_for_scaling()
  reflections = PrfIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.prf.value']) == [1.0, 3.0, 4.0, 20.0, 3.0]
  assert list(reflections['intensity.prf.variance']) == pytest.approx(
    [0.1, 0.3, 0.4, 0.5*16.0, 0.6*0.25])

  #check it still passes with no partiality correction
  reflections = generate_test_reflections_for_scaling()
  del reflections['partiality']
  reflections = PrfIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.prf.value']) == [
    1.0, 2.0, 3.0, 4.0, 20.0, 3.0]

  #test IsgiI selecting
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0])
  r['intensity.prf.variance'] = flex.double([1.0, 1.0])
  r = PrfIntensityReducer.filter_on_min_isigi(r, 1.5)
  assert list(r['intensity.prf.value']) == [2.0]

  # now test a typical case - reflection table with inv scale factor,
  # check that we only apply that correction and do any relevant filtering
  reflections = generate_integrated_test_reflections()
  reflections['qe'] = flex.double(6, 2)
  reflections = PrfIntensityReducer.filter_for_export(reflections)

  assert list(reflections['intensity.prf.value']) == pytest.approx(
    [1.0/2.0, 2.0/2.0, 3.0/2.0, 6.0/2.0])
  assert list(reflections['intensity.prf.variance']) == pytest.approx(
    [0.1/4.0, 0.2/4.0, 0.3/4.0, 0.6/4.0])

  assert list(reflections['fractioncalc']) == [1.0] * 4

  assert not 'intensity.sum.value' in reflections
  assert not 'intensity.scale.value' in reflections
  assert not 'intensity.sum.variance' in reflections
  assert not 'intensity.scale.variance' in reflections

def test_SumIntensityReducer():
  """Test the methods of the SumIntensityReducer class"""
  reflections = generate_integrated_test_reflections()
  reflections = SumIntensityReducer.reduce_on_intensities(reflections)
  #reflections = reduce_on_summation_intensities(reflections)
  assert list(reflections['intensity.sum.value']) == [11.0, 12.0, 13.0, 14.0]
  assert list(reflections['intensity.sum.variance']) == [1.1, 1.2, 1.3, 1.4]

  reflections = generate_test_reflections_for_scaling()
  reflections = SumIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.sum.value']) == [110.0, 13.0, 14.0, 60.0, 8.0]
  assert list(reflections['intensity.sum.variance']) == pytest.approx(
    [110.0, 1.3, 1.4, 24.0, 0.4])

  #check it still passes with no partiality correction
  reflections = generate_test_reflections_for_scaling()
  del reflections['partiality']
  reflections = SumIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.sum.value']) == [
    11.0, 12.0, 13.0, 14.0, 60.0, 8.0]

  #test IsgiI selecting
  r = flex.reflection_table()
  r['intensity.sum.value'] = flex.double([1.0, 2.0])
  r['intensity.sum.variance'] = flex.double([1.0, 1.0])
  r = SumIntensityReducer.filter_on_min_isigi(r, 1.5)
  assert list(r['intensity.sum.value']) == [2.0]

  # now test a typical case - reflection table with inv scale factor,
  # check that we only apply that correction and do any relevant filtering
  reflections = generate_integrated_test_reflections()
  reflections['lp'] = flex.double(6, 2)
  reflections['partiality'] = flex.double([1.0, 1.0, 0.5, 0.1, 1.0, 1.0])
  reflections = SumIntensityReducer.filter_for_export(reflections,
    partiality_threshold=0.25)

  assert list(reflections['intensity.sum.value']) == pytest.approx(
    [22.0, 24.0, 52.0])
  assert list(reflections['intensity.sum.variance']) == pytest.approx(
    [4.4, 4.8, 20.8])

  assert not 'intensity.prf.value' in reflections
  assert not 'intensity.scale.value' in reflections
  assert not 'intensity.prf.variance' in reflections
  assert not 'intensity.scale.variance' in reflections

def test_SumAndPrfIntensityReducer():
  """Test that the reflection table is reduced on prf and sum intensities"""
  reflections = generate_integrated_test_reflections()
  reflections = SumAndPrfIntensityReducer.reduce_on_intensities(reflections)
  assert list(reflections['intensity.prf.value']) == [1.0, 2.0, 3.0]
  assert list(reflections['intensity.prf.variance']) == [0.1, 0.2, 0.3]
  assert list(reflections['intensity.sum.value']) == [11.0, 12.0, 13.0]
  assert list(reflections['intensity.sum.variance']) == [1.1, 1.2, 1.3]

  reflections['lp'] = flex.double([0.5, 0.5, 1.0])
  reflections['partiality'] = flex.double([0.0, 1.0, 1.0])
  reflections = SumAndPrfIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.prf.value']) == [1.0, 3.0]
  assert list(reflections['intensity.prf.variance']) == [0.2*0.25, 0.3]
  assert list(reflections['intensity.sum.value']) == [6.0, 13.0]
  assert list(reflections['intensity.sum.variance']) == [0.3, 1.3]

  #test IsgiI selecting
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0, 3.0])
  r['intensity.prf.variance'] = flex.double([1.0, 1.0, 1.0])
  r['intensity.sum.value'] = flex.double([1.0, 2.0, 3.0])
  r['intensity.sum.variance'] = flex.double([1.0, 1.1, 1.0])
  r = SumAndPrfIntensityReducer.filter_on_min_isigi(r, 1.999)
  assert list(r['intensity.prf.value']) == [3.0]

  #test bad variance filtering
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0, 3.0])
  r['intensity.prf.variance'] = flex.double([0.0, 1.0, 1.0])
  r['intensity.sum.value'] = flex.double([1.0, 2.0, 3.0])
  r['intensity.sum.variance'] = flex.double([1.0, 0.0, 1.0])
  r = SumAndPrfIntensityReducer.filter_bad_variances(r)
  assert list(r['intensity.prf.value']) == [3.0]

  # test filtering for export
  reflections = generate_integrated_test_reflections()
  reflections = SumAndPrfIntensityReducer.filter_for_export(reflections)
  assert not 'intensity.scale.value' in reflections
  assert not 'intensity.scale.variance' in reflections

def test_ScaleIntensityReducer():
  """Test that the reflection table is reduced on scaling intensities"""
  reflections = generate_integrated_test_reflections()
  reflections = ScaleIntensityReducer.reduce_on_intensities(reflections)
  assert list(reflections['intensity.scale.value']) == pytest.approx(
    [23.0, 24.0, 25.0])
  assert list(reflections['intensity.scale.variance']) == pytest.approx(
    [2.3, 2.4, 2.5])
  del reflections['inverse_scale_factor']
  with pytest.raises(Sorry):
    reflections = ScaleIntensityReducer.reduce_on_intensities(reflections)

  reflections = generate_test_reflections_for_scaling()
  reflections = ScaleIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.scale.value']) == pytest.approx(
    [21.0/5.0, 23.0/5.0, 24.0/10.0, 25.0/10.0, 26.0/10.0, 27.0/10.0])
  assert list(reflections['intensity.scale.variance']) == pytest.approx(
    [2.1/25.0, 2.3/25.0, 2.4/100, 2.5/100, 2.6/100, 2.7/100])
  del reflections['inverse_scale_factor']
  with pytest.raises(Sorry):
    reflections = ScaleIntensityReducer.apply_scaling_factors(reflections)

  reflections = generate_test_reflections_for_scaling()
  del reflections['partiality']
  reflections = ScaleIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.scale.value']) == [
    21.0/5.0, 22.0/5.0, 23.0/5.0, 24.0/10.0, 25.0/10.0, 26.0/10.0, 27.0/10.0]

  #test IsgiI selecting
  r = flex.reflection_table()
  r['intensity.scale.value'] = flex.double([1.0, 2.0])
  r['intensity.scale.variance'] = flex.double([1.0, 1.0])
  r = ScaleIntensityReducer.filter_on_min_isigi(r, 1.5)
  assert list(r['intensity.scale.value']) == [2.0]

  # now test a typical case - reflection table with inv scale factor,
  # check that we only apply that correction and do any relevant filtering
  reflections = generate_integrated_test_reflections()
  reflections['lp'] = flex.double(6, 0.6)
  reflections = ScaleIntensityReducer.filter_for_export(reflections)

  assert list(reflections['intensity.scale.value']) == pytest.approx(
    [23.0/5.0, 24.0/10.0, 25.0/10.0])
  assert list(reflections['intensity.scale.variance']) == pytest.approx(
    [2.3/25.0, 0.024, 0.025])

  assert not 'intensity.prf.value' in reflections
  assert not 'intensity.sum.value' in reflections
  assert not 'intensity.prf.variance' in reflections
  assert not 'intensity.sum.variance' in reflections

def test_AllSumPrfScaleIntensityReducer():
  reflections = generate_integrated_test_reflections()
  reflections = AllSumPrfScaleIntensityReducer.reduce_on_intensities(reflections)
  assert list(reflections['intensity.scale.value']) == [23.0]
  assert list(reflections['intensity.scale.variance']) == [2.3]
  assert list(reflections['intensity.prf.value']) == [3.0]
  assert list(reflections['intensity.prf.variance']) == [0.3]
  assert list(reflections['intensity.sum.value']) == [13.0]
  assert list(reflections['intensity.sum.variance']) == [1.3]
  reflections = AllSumPrfScaleIntensityReducer.apply_scaling_factors(reflections)
  assert list(reflections['intensity.scale.value']) == [23.0/5.0]
  assert list(reflections['intensity.scale.variance']) == [2.3/25.0]
  assert list(reflections['intensity.prf.value']) == [3.0]
  assert list(reflections['intensity.prf.variance']) == [0.3]
  assert list(reflections['intensity.sum.value']) == [13.0]
  assert list(reflections['intensity.sum.variance']) == [1.3]

def test_filter_reflection_table():
  """Test the interface function"""
  reflections = generate_integrated_test_reflections()
  reflections = filter_reflection_table(reflections, ['sum'])
  assert  'intensity.sum.value' in reflections
  assert not 'intensity.prf.value' in reflections
  assert not 'intensity.scale.value' in reflections
  reflections = generate_integrated_test_reflections()
  reflections = filter_reflection_table(reflections, ['prf'])
  assert  'intensity.prf.value' in reflections
  assert not 'intensity.sum.value' in reflections
  assert not 'intensity.scale.value' in reflections
  reflections = generate_integrated_test_reflections()
  reflections = filter_reflection_table(reflections, ['scale'])
  assert  'intensity.scale.value' in reflections
  assert not 'intensity.prf.value' in reflections
  assert not 'intensity.sum.value' in reflections
  reflections = generate_integrated_test_reflections()
  reflections = filter_reflection_table(reflections, ['sum', 'prf'])
  assert  'intensity.sum.value' in reflections
  assert  'intensity.prf.value' in reflections
  assert not 'intensity.scale.value' in reflections
  reflections = generate_integrated_test_reflections()
  reflections = filter_reflection_table(reflections, ['sum', 'prf', 'scale'])
  assert 'intensity.sum.value' in reflections
  assert 'intensity.prf.value' in reflections
  assert 'intensity.scale.value' in reflections
  with pytest.raises(Sorry):
    reflections = filter_reflection_table(reflections, ['bad'])

def return_reflections_side_effect(reflections, *args, **kwargs):
  """Side effect for overriding the call to reject_outliers."""
  return reflections

def test_checks_in_reduce_data_for_export():
  """Test for graceful failures if serious problems with the data are found."""
  # If no valid ids, should raise sorry
  r = flex.reflection_table()
  r['id'] = flex.int([-1, -1, -1])
  r['intensity.prf.value'] = flex.double(3, 1.0)
  r['intensity.prf.variance'] = flex.double(3, 1.0)
  with pytest.raises(Sorry):
    r = PrfIntensityReducer.filter_for_export(r)

  # If no valid prf, should raise sorry
  r = flex.reflection_table()
  r['id'] = flex.int([0, 0])
  r['intensity.prf.value'] = flex.double([1.0, 2.0])
  r['intensity.prf.variance'] = flex.double([0.0, 1.0])
  r.set_flags(flex.bool([False, False]), r.flags.integrated_prf)
  with pytest.raises(Sorry):
    r = PrfIntensityReducer.filter_for_export(r)
  r.set_flags(flex.bool([True, True]), r.flags.integrated_prf)

  # Should filter bad variances
  r = PrfIntensityReducer.filter_bad_variances(r)
  assert r.size() == 1

  # If reflection table is empty
  r = flex.reflection_table()
  with pytest.raises(AssertionError):
    r = PrfIntensityReducer.filter_for_export(r)

  # What if all ice ring
  r = generate_simple_table()
  r.set_flags(flex.bool([True, True, True]), r.flags.in_powder_ring)
  r = PrfIntensityReducer.filter_for_export(r,
        filter_ice_rings=True, min_isigi=1.0, partiality_threshold=0.99)
  assert r.size() == 0 #Would we want it to raise a Sorry before this?

  # What if none left on SigI
  r = generate_simple_table()
  r = PrfIntensityReducer.filter_for_export(r,
        filter_ice_rings=False, min_isigi=4.0, partiality_threshold=0.99)
  assert r.size() == 0 #Would we want it to raise a Sorry before this?

def test_partial_summing_functions():
  """These functions set the first entry to be the sum/weighted sum for the
  partials. Example is that of two reflections measured with 0.5 partiality,
  with I/sigma of 1 & 2."""
  #FIXME: What weights should be used
  #Test combining scale values
  r = flex.reflection_table()
  r['intensity.scale.value'] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0])
  r['intensity.scale.variance'] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0])
  partials_list = [0, 1]
  r = _sum_scale_partials(r, partials_list)
  assert r.size() == 5
  assert r['intensity.scale.value'][0] == 1.5
  assert r['intensity.scale.variance'][0] == 0.5

  #Test combining prf values
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0])
  r['intensity.prf.variance'] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0])
  partials_list = [0, 1]
  r = _sum_prf_partials(r, partials_list)
  assert r.size() == 5
  assert r['intensity.prf.value'][0] == 1.8
  assert r['intensity.prf.variance'][0] == 1.0

  #Test summing sum values
  # "Same data - sum values should be half, adjust variances accordingly
  # to make I/sigma = 1,2"
  r = flex.reflection_table()
  r['intensity.sum.value'] = flex.double([0.5, 1.0, 3.0, 4.0, 5.0])
  r['intensity.sum.variance'] = flex.double([0.25, 0.25, 1.0, 1.0, 1.0])
  partials_list = [0, 1]
  r = _sum_sum_partials(r, partials_list)
  assert r.size() == 5
  assert r['intensity.sum.value'][0] == 1.5
  assert r['intensity.sum.variance'][0] == 0.5

def test_sum_partial_reflections():
  """Test the partial summing function for the different intensity types."""
  r = flex.reflection_table()
  r['intensity.prf.value'] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0])
  r['intensity.prf.variance'] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0])
  r['partial_id'] = flex.int([0, 0, 1, 1, 2])
  r['partiality'] = flex.double([0.5, 0.4, 0.6, 0.2, 0.9])
  r['identifier'] = flex.int([1, 2, 3, 4, 5])

  r = sum_partial_reflections(r)
  assert list(r['identifier']) == [1, 3, 5]
  assert list(r['partiality']) == [0.9, 0.8, 0.9]

  r = flex.reflection_table()
  r['intensity.sum.value'] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0])
  r['intensity.sum.variance'] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0])
  r['partial_id'] = flex.int([0, 0, 1, 1, 2])
  r['partiality'] = flex.double([0.5, 0.4, 0.6, 0.2, 0.9])
  r['identifier'] = flex.int([1, 2, 3, 4, 5])

  r = sum_partial_reflections(r)
  assert list(r['identifier']) == [1, 3, 5]
  assert list(r['partiality']) == [0.9, 0.8, 0.9]

  r = flex.reflection_table()
  r['intensity.scale.value'] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0])
  r['intensity.scale.variance'] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0])
  r['partial_id'] = flex.int([0, 0, 1, 1, 2])
  r['partiality'] = flex.double([0.5, 0.4, 0.6, 0.2, 0.9])
  r['identifier'] = flex.int([1, 2, 3, 4, 5])

  r = sum_partial_reflections(r)
  assert list(r['identifier']) == [1, 3, 5]
  assert list(r['partiality']) == [0.9, 0.8, 0.9]

  #if all partiality of one - should just return same
  r = flex.reflection_table()
  r['intensity.scale.value'] = flex.double([1.0, 2.0, 3.0])
  r['intensity.scale.variance'] = flex.double([1.0, 1.0, 1.0])
  r['partiality'] = flex.double(3, 1.0)
  r['identifier'] = flex.int([1, 2, 3])
  assert list(r['identifier']) == [1, 2, 3]

  # Add test to check calculation in case where both prf and sum - but this
  # requires knowing how the values will be weighted, so leave until that is
  # decided.