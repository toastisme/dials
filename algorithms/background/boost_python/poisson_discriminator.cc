/*
 * poisson_discriminator.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/background/poisson_discriminator.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  void export_poisson_discriminator()
  {
    def("moment", &moment, (
      arg("data"), arg("centre"), arg("k")));
  
    // Export normality test
    def("is_poisson_distributed", &is_poisson_distributed, (
      arg("data"), arg("n_sigma")));  
    
    // Overloads for call method
    void (PoissonDiscriminator::*call_shoebox_and_mask)(const flex_int&,
        flex_int &) const = &PoissonDiscriminator::operator();
    flex_int (PoissonDiscriminator::*call_shoebox)(const flex_int&) const =
      &PoissonDiscriminator::operator();
  
    class_<PoissonDiscriminator, bases<DiscriminatorStrategy> >(
        "PoissonDiscriminator")
      .def(init<std::size_t, double>((
        arg("min_data") = 10,
        arg("n_sigma") = 3.0)))
      .def("__call__", call_shoebox_and_mask, (
        arg("shoebox"),
        arg("mask")))
      .def("__call__", call_shoebox, (
        arg("shoebox")));
  }

}}} // namespace = dials::algorithms::boost_python
