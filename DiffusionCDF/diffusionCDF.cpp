#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <math.h>

#include "pybind11_numpy_scalar.h"
#include "diffusionCDF.hpp"
#include "../Stats/stat.h"

namespace py = pybind11;

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

// Boilerplate to get PyBind11 to cast to a npquad precision.
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

DiffusionCDF::DiffusionCDF(const double _beta, const unsigned long int _tMax)
{
  beta = _beta;
  tMax = _tMax;

  if (_beta != 0) {
    boost::random::beta_distribution<>::param_type params(_beta, _beta);
    betaParams = params;
  }

  std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
  dis.param(unifParams);
  gen.seed(rd());
}

double DiffusionCDF::generateBeta()
{
  // If beta = 0 return either 0 or 1
  if (beta == 0.0) {
    return round(dis(gen));
  }
  // If beta = 1 use random uniform distribution
  else if (beta == 1) {
    return dis(gen);
  }
  // If beta = inf return 0.5
  else if (isinf(beta)) {
    return 0.5;
  }
  else {
    return beta_dist(gen, betaParams);
  }
}

DiffusionTimeCDF::DiffusionTimeCDF(const double _beta, const unsigned long int _tMax) : DiffusionCDF(_beta, _tMax)
{
  CDF.resize(tMax + 1);
  CDF[0] = 1;
}

void DiffusionTimeCDF::iterateTimeStep()
{
  std::vector<RealType> CDF_next(tMax+1);
  for (unsigned long int n = 0; n <= t+1; n++){
    if (n == 0){
      CDF_next[n] = 1; // Need CDF(n=0, t) = 1
    }
    else if (n == t+1){
      RealType beta = RealType(generateBeta());
      CDF_next[n] = beta * CDF[n-1];
    }
    else{
      /* Maybe add this in
      if (CDF[n-1] == CDF[n]){
        continue // or could even break?
      }
      */
      RealType beta = RealType(generateBeta());
      CDF_next[n] = beta * CDF[n-1] + (1 - beta) * CDF[n];
    }
  }
  CDF = CDF_next;
  t += 1;
}

unsigned long int DiffusionTimeCDF::findQuantile(RealType quantile)
{
  unsigned long int quantilePosition;
  for (unsigned long int n = t; n >= 0; n--){
    if (CDF[n] > 1 / quantile){
      quantilePosition = 2* n + 2 - t;
      break;
    }
  }
  return quantilePosition;
}

unsigned long int DiffusionTimeCDF::findLowerQuantile(RealType quantile) {
  long int quantilePosition=0;
  for (unsigned long int n = 0; n <= t; n++) {
    if (CDF[n] < 1. - 1 / quantile) {
      quantilePosition = 2 * n - t; 
      break; 
    }
  }
  return quantilePosition;
}

std::vector<unsigned long int> DiffusionTimeCDF::findQuantiles(
  std::vector<RealType> quantiles)
{
  // Sort incoming quantiles b/c we need them to be in descending order for
  // algorithm to work
  std::sort(quantiles.begin(), quantiles.end(), std::greater<RealType>());

  // Initialize to have same number of vectors as in Ns
  std::vector<unsigned long int> quantilePositions(quantiles.size());
  unsigned long int quantile_idx = 0;
  for (unsigned long int n = t; n >= 0; n--){
    while(CDF[n] > 1 / quantiles[quantile_idx]){
      quantilePositions[quantile_idx] = 2 * n + 2 - t;
      quantile_idx += 1;

      //Break while loop if past last position
      if (quantile_idx == quantiles.size()){
        break;
      }
    }
    // Also need to break for loop if in last position b/c we are done searching
    if (quantile_idx == quantiles.size()){
      break;
    }
  }
  return quantilePositions;
}

std::vector<long int> DiffusionTimeCDF::getxvals(){
  std::vector<long int> xvals(t+1);
  for (unsigned int n = 0; n < xvals.size(); n++){
    xvals[n] = 2 * n - t;
  }
  return xvals;
}

RealType DiffusionTimeCDF::getGumbelVariance(RealType nParticles)
{
  std::vector<RealType> cdf = slice(CDF, 0, t);
  cdf.push_back(0); // Need to add 0 to CDF to make it complete.

  std::vector<long int> xvals = getxvals();
  RealType var = getGumbelVarianceCDF(xvals, cdf, nParticles);
  return var;
}

std::vector<RealType> DiffusionTimeCDF::getGumbelVariance(std::vector<RealType> nParticles){
  std::vector<RealType> cdf = slice(CDF, 0, t);
  cdf.push_back(0); // Need to add 0 to CDF to make it complete.
  std::vector<long int> xvals = getxvals();
  return getGumbelVarianceCDF(xvals, cdf, nParticles);
}

std::pair<RealType, float> DiffusionTimeCDF::getProbandV(RealType quantile){
  unsigned long int quantilePosition;
  RealType prob;
  for (unsigned long int n = t; n >= 0; n--){
    if (CDF[n] > 1 / quantile){
      quantilePosition = 2 * n - t;
      prob = CDF[n];
      break;
    }
  }
  float v = (float)quantilePosition / (float)t;
  return std::make_pair(prob, v);
}


std::vector<RealType> DiffusionTimeCDF::getSaveCDF(){
  return slice(CDF, 0, t);
}

DiffusionPositionCDF::DiffusionPositionCDF(const double _beta, const unsigned long int _tMax, std::vector<RealType> _quantiles) : DiffusionCDF(_beta, _tMax)
{
  // Initialize such that CDF(n=0, t) = 1
  CDF.resize(tMax+1, 1);
  quantiles = _quantiles;

  // Create a measurment array such that each column is a different quantile
  // measurement and each row is a differeent time.
  quantilePositions.resize(quantiles.size());
  for (unsigned long int i=0; i < quantilePositions.size(); i++){
    quantilePositions[i].resize(tMax+1, 0);
    // At t=0, all the quantiles will be 2 (2*n + 2 - t for n=t=0) - I think this is mainly for testing.
    quantilePositions[i][0] = 2;
  }
}

void DiffusionPositionCDF::stepPosition()
{
  // Initialize all values to 0 since below CDF(n, n) is 0
  std::vector<RealType> CDF_next(tMax+1, 0);

  for (unsigned long int t = position+1; t < tMax + 1; t++){
    RealType beta = RealType(generateBeta());
    if (t == position+1){
      CDF_next[t] = beta * CDF[t-1];
    }
    else{
      CDF_next[t] = beta * CDF[t-1] + (1 - beta) * CDF_next[t-1];
    }

    // Now want to loop through every quantile vector and update positions of
    // each quantile at the current time.
    for (unsigned long int i=0; i < quantiles.size(); i++){
      if (CDF_next[t] > 1 / quantiles[i]){
        // quantilePositions[i][t] is the ith quantile in the list at time t
        // Note: position will always be greater so don't need to check this.
        quantilePositions[i][t] = 2 * (position + 1) + 2 - t;
      }
    }
  }
  CDF = CDF_next;
  position += 1;
}


PYBIND11_MODULE(diffusionCDF, m)
{
  m.doc() = "Diffusion recurrance relation";
  py::class_<DiffusionCDF>(m, "DiffusionCDF")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getBeta", &DiffusionCDF::getBeta)
      .def("getCDF", &DiffusionCDF::getCDF)
      .def("setCDF", &DiffusionCDF::setCDF, py::arg("CDF"))
      .def("gettMax", &DiffusionCDF::gettMax)
      .def("settMax", &DiffusionCDF::settMax)
      .def("setBetaSeed", &DiffusionCDF::setBetaSeed, py::arg("seed"));

  py::class_<DiffusionTimeCDF, DiffusionCDF>(m, "DiffusionTimeCDF")
      .def(py::init<const double, const unsigned long int>(), py::arg("beta"), py::arg("tMax"))
      .def("getGumbelVariance", static_cast<RealType (DiffusionTimeCDF::*)(RealType)>(&DiffusionTimeCDF::getGumbelVariance), py::arg("nParticles"))
      .def("getGumbelVariance", static_cast<std::vector<RealType> (DiffusionTimeCDF::*)(std::vector<RealType>)>(&DiffusionTimeCDF::getGumbelVariance), py::arg("nParticles"))
      .def("getTime", &DiffusionTimeCDF::getTime)
      .def("setTime", &DiffusionTimeCDF::setTime)
      .def("iterateTimeStep", &DiffusionTimeCDF::iterateTimeStep)
      .def("findQuantile", &DiffusionTimeCDF::findQuantile, py::arg("quantile"))
      .def("findQuantiles", &DiffusionTimeCDF::findQuantiles, py::arg("quantiles"))
      .def("findLowerQuantile", &DiffusionTimeCDF::findLowerQuantile, py::arg("quantile"))
      .def("getSaveCDF", &DiffusionTimeCDF::getSaveCDF)
      .def("getxvals", &DiffusionTimeCDF::getxvals)
      .def("getProbandV", &DiffusionTimeCDF::getProbandV, py::arg("quantile"));

  py::class_<DiffusionPositionCDF, DiffusionCDF>(m, "DiffusionPositionCDF")
      .def(py::init<const double, const unsigned long int, std::vector<RealType> >(), py::arg("beta"), py::arg("tMax"), py::arg("quantiles"))
      .def("getPosition", &DiffusionPositionCDF::getPosition)
      .def("getQuantilePositions", &DiffusionPositionCDF::getQuantilePositions)
      .def("getQuantiles", &DiffusionPositionCDF::getQuantiles)
      .def("stepPosition", &DiffusionPositionCDF::stepPosition);
}
