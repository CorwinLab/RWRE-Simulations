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

typedef boost::multiprecision::float128 RealType;

#ifndef DIFFUSIONCDF_HPP_
#define DIFFUSIONCDF_HPP_

// Base Diffusion class
class DiffusionCDF{
protected:
  std::vector<RealType> CDF;
  double beta;
  unsigned long int tMax;

  // It would be nice if this could be a generic distribution as:
  // boost::random::distribution bias
  boost::random::beta_distribution<>::param_type betaParams;

  std::random_device rd;
  boost::random::mt19937_64 gen;

  std::uniform_real_distribution<> dis;
  boost::random::beta_distribution<> beta_dist;

  double generateBeta();

public:
  DiffusionCDF(const double _beta, const unsigned long int _tMax);
  ~DiffusionCDF(){};

  double getBeta() { return beta; };
  void setBeta(double _beta){ beta = _beta; };

  std::vector<RealType> getCDF() { return CDF; };
  void setCDF(std::vector<RealType> _CDF){ CDF = _CDF; };

  unsigned long int gettMax() { return tMax; };
  void settMax(unsigned long int _tMax){ tMax = _tMax; };

  void setBetaSeed(const unsigned int seed) { gen.seed(seed); };

};

class DiffusionTimeCDF: public DiffusionCDF {
private:
  unsigned long int t = 0;

public:
  DiffusionTimeCDF(const double _beta, const unsigned long int _tMax);

  unsigned long int getTime() { return t; };
  void setTime(unsigned long int _t){ t = _t; };

  // Functions that do things
  void iterateTimeStep();

  unsigned long int findQuantile(RealType quantile);
  std::vector<unsigned long int> findQuantiles(std::vector<RealType> quantiles);

  unsigned long int findLowerQuantile(RealType quantile);


  RealType getGumbelVariance(RealType nParticles);
  std::vector<RealType> getGumbelVariance(std::vector<RealType> nParticles);
  std::vector<long int> getxvals();
  std::vector<RealType> getSaveCDF();
  std::pair<RealType, float> getProbandV(RealType quantile);
};

class DiffusionPositionCDF: public DiffusionCDF{
private:
  unsigned long int position = 0;
  std::vector<RealType> quantiles;
  std::vector<std::vector<unsigned long int> > quantilePositions;

public:
  DiffusionPositionCDF(const double _beta, const unsigned long int _tMax, std::vector<RealType> _quantiles);
  unsigned long int getPosition() { return position; };
  std::vector<std::vector<unsigned long int> > getQuantilePositions() { return quantilePositions; };
  std::vector<RealType> getQuantiles() { return quantiles; };

  // Functions that do things
  void stepPosition();
};

#endif /* DIFFUSIONCDF_HPP_ */
