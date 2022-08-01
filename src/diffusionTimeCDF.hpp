#ifndef DIFFUSIONTIMECDF_HPP_
#define DIFFUSIONTIMECDF_HPP_

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

// Base Diffusion class
class DiffusionTimeCDF : public DiffusionCDF {
private:
  unsigned long int t = 0;

public:
  DiffusionTimeCDF(const double _beta, const unsigned long int _tMax);

  unsigned long int getTime() { return t; };
  void setTime(unsigned long int _t) { t = _t; };

  // Functions that do things
  void iterateTimeStep();

  long int findQuantile(RealType quantile);
  std::vector<long int> findQuantiles(std::vector<RealType> quantiles);

  long int findLowerQuantile(RealType quantile);

  RealType getGumbelVariance(RealType nParticles);
  std::vector<RealType> getGumbelVariance(std::vector<RealType> nParticles);
  std::vector<long int> getxvals();
  std::vector<RealType> getSaveCDF();
  std::pair<RealType, float> getProbandV(RealType quantile);
};

#endif /* DIFFUSIONTIMECDF_HPP_ */
