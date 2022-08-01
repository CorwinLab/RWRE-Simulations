#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "randomNumGenerator.hpp"
#include "diffusionCDFBase.hpp"
#include "diffusionTimeCDF.hpp"
#include "diffusionPositionCDF.hpp"
#include "diffusionPDF.hpp"
#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 256;

// Kinda following:
// https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <> struct npy_format_descriptor<RealType> {
  static constexpr auto name = _("RealType");
  static pybind11::dtype dtype()
  {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <> struct type_caster<RealType> : npy_scalar_caster<RealType> {
  static constexpr auto name = _("RealType");
};

} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(libDiffusion, m)
{
     m.doc() = "Random walk library";
     py::class_<RandomNumGenerator>(m, "RandomNumGenerator")
          .def(py::init<const double>())
          .def("getBeta", &RandomNumGenerator::getBeta)
          .def("setBeta", &RandomNumGenerator::setBeta)
          .def("generateBeta", &RandomNumGenerator::generateBeta)
          .def("setBetaSeed", &RandomNumGenerator::setBetaSeed);
          
     py::class_<DiffusionPDF, RandomNumGenerator>(m, "DiffusionPDF")
          .def(py::init<const RealType,
                         const double,
                         const unsigned long int,
                         const bool,
                         const bool>(),
               py::arg("numberOfParticles"),
               py::arg("beta"),
               py::arg("occupancySize"),
               py::arg("ProbDistFlag") = true,
               py::arg("staticEnvironment") = false)
          .def("getOccupancy", &DiffusionPDF::getOccupancy)
          .def("setOccupancy", &DiffusionPDF::setOccupancy, py::arg("occupancy"))
          .def("getOccupancySize", &DiffusionPDF::getOccupancySize)
          .def("getTransitionProbabilities", &DiffusionPDF::getTransitionProbabilities)
          .def("getSaveOccupancy", &DiffusionPDF::getSaveOccupancy)
          .def("getStaticEnvironment", &DiffusionPDF::getStaticEnvironment)
          .def("setStaticEnvironment", &DiffusionPDF::setStaticEnvironment)
          .def("resizeOccupancy",
               &DiffusionPDF::resizeOccupancy,
               py::arg("size"))
          .def("getNParticles", &DiffusionPDF::getNParticles)
          .def("getBeta", &DiffusionPDF::getBeta)
          .def("setProbDistFlag",
               &DiffusionPDF::setProbDistFlag,
               py::arg("ProbDistFlag"))
          .def("getProbDistFlag", &DiffusionPDF::getProbDistFlag)
          .def("getSmallCutoff", &DiffusionPDF::getSmallCutoff)
          .def("setSmallCutoff",
               &DiffusionPDF::setSmallCutoff,
               py::arg("smallCutoff"))
          .def("getLargeCutoff", &DiffusionPDF::getLargeCutoff)
          .def("setLargeCutoff",
               &DiffusionPDF::setLargeCutoff,
               py::arg("largeCutoff"))
          .def("getEdges", &DiffusionPDF::getEdges)
          .def("setEdges", &DiffusionPDF::setEdges)
          .def("getMaxIdx", &DiffusionPDF::getMaxIdx)
          .def("getMinIdx", &DiffusionPDF::getMinIdx)
          .def("getTime", &DiffusionPDF::getTime)
          .def("setTime", &DiffusionPDF::setTime)
          .def("iterateTimestep", &DiffusionPDF::iterateTimestep)
          .def("findQuantile", &DiffusionPDF::findQuantile, py::arg("quantile"))
          .def("findQuantiles", &DiffusionPDF::findQuantiles, py::arg("quantiles"))
          .def("pGreaterThanX", &DiffusionPDF::pGreaterThanX, py::arg("idx"))
          .def("calcVsAndPb", &DiffusionPDF::calcVsAndPb, py::arg("num"))
          .def("VsAndPb", &DiffusionPDF::VsAndPb, py::arg("v"))
          .def("getGumbelVariance",
               &DiffusionPDF::getGumbelVariance,
               py::arg("nParticles"))
          .def("getCDF", &DiffusionPDF::getCDF);
     
     py::class_<DiffusionCDF, RandomNumGenerator>(m, "DiffusionCDF")
          .def(py::init<const double, const unsigned long int>(),
               py::arg("beta"),
               py::arg("tMax"))
          .def("getBeta", &DiffusionCDF::getBeta)
          .def("getCDF", &DiffusionCDF::getCDF)
          .def("setCDF", &DiffusionCDF::setCDF, py::arg("CDF"))
          .def("gettMax", &DiffusionCDF::gettMax)
          .def("settMax", &DiffusionCDF::settMax)
          .def("setBetaSeed", &DiffusionCDF::setBetaSeed, py::arg("seed"));

     py::class_<DiffusionTimeCDF, DiffusionCDF>(m, "DiffusionTimeCDF")
          .def(py::init<const double, const unsigned long int>(),
               py::arg("beta"),
               py::arg("tMax"))
          .def("getGumbelVariance",
               static_cast<RealType (DiffusionTimeCDF::*)(RealType)>(
                    &DiffusionTimeCDF::getGumbelVariance),
               py::arg("nParticles"))
          .def("getGumbelVariance",
               static_cast<std::vector<RealType> (DiffusionTimeCDF::*)(
                    std::vector<RealType>)>(&DiffusionTimeCDF::getGumbelVariance),
               py::arg("nParticles"))
          .def("getTime", &DiffusionTimeCDF::getTime)
          .def("setTime", &DiffusionTimeCDF::setTime)
          .def("iterateTimeStep", &DiffusionTimeCDF::iterateTimeStep)
          .def("findQuantile", &DiffusionTimeCDF::findQuantile, py::arg("quantile"))
          .def("findQuantiles",
               &DiffusionTimeCDF::findQuantiles,
               py::arg("quantiles"))
          .def("findLowerQuantile",
               &DiffusionTimeCDF::findLowerQuantile,
               py::arg("quantile"))
          .def("getSaveCDF", &DiffusionTimeCDF::getSaveCDF)
          .def("getxvals", &DiffusionTimeCDF::getxvals)
          .def("getProbandV", &DiffusionTimeCDF::getProbandV, py::arg("quantile"))
          .def("generateBeta", &DiffusionTimeCDF::generateBeta);

     py::class_<DiffusionPositionCDF, DiffusionCDF>(m, "DiffusionPositionCDF")
          .def(py::init<const double,
                         const unsigned long int,
                         std::vector<RealType>>(),
               py::arg("beta"),
               py::arg("tMax"),
               py::arg("quantiles"))
          .def("getPosition", &DiffusionPositionCDF::getPosition)
          .def("getQuantilePositions", &DiffusionPositionCDF::getQuantilePositions)
          .def("getQuantiles", &DiffusionPositionCDF::getQuantiles)
          .def("stepPosition", &DiffusionPositionCDF::stepPosition);
}
