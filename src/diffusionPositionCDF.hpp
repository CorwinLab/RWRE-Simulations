#ifndef DIFFUSIONPOSITIONCDF_HPP_
#define DIFFUSIONPOSITIONCDF_HPP_

#include <assert.h>
#include <boost/math/distributions.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/random.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <math.h>
#include <random>
#include <utility>
#include <vector>
#include "diffusionCDFBase.hpp"
#include "randomNumGenerator.hpp"

typedef boost::multiprecision::float128 RealType;

class DiffusionPositionCDF : public DiffusionCDF {
private:
  unsigned long int position = 0;
  std::vector<RealType> quantiles;
  std::vector<std::vector<unsigned long int>> quantilePositions;

public:
  DiffusionPositionCDF(const double _beta,
                       const unsigned long int _tMax,
                       std::vector<RealType> _quantiles);
  unsigned long int getPosition() { return position; };
  std::vector<std::vector<unsigned long int>> getQuantilePositions()
  {
    return quantilePositions;
  };
  std::vector<RealType> getQuantiles() { return quantiles; };

  // Functions that do things
  void stepPosition();
};

#endif /* DIFFUSIONPOSITIONCDF_HPP_ */
