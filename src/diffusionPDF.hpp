#ifndef DIFFUSIONPDF_HPP_
#define DIFFUSIONPDF_HPP_

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
#include "randomNumGenerator.hpp"

typedef boost::multiprecision::float128 RealType;

class DiffusionPDF : public RandomNumGenerator {
private:
  std::vector<RealType> occupancy;
  RealType nParticles;
  unsigned long int occupancySize;
  bool ProbDistFlag;
  bool staticEnvironment;
  std::vector<RealType> transitionProbabilities;
  double smallCutoff = pow(2, 31) - 2;
  double largeCutoff = 1e64;

  boost::random::binomial_distribution<> binomial;

  std::pair<unsigned long int, unsigned long int>
      edges;
  unsigned long int time;

  RealType toNextSite(RealType currentSite, RealType bias);
  double generateBeta();

public:
  DiffusionPDF(const RealType _nParticles,
               const double _beta,
               const unsigned long int _occupancySize,
               const bool _probDistFlag = true,
               const bool _staticEnvironment = false);
  ~DiffusionPDF(){};

  RealType getNParticles() { return nParticles; };

  void setProbDistFlag(bool _probDistFlag) { ProbDistFlag = _probDistFlag; };
  bool getProbDistFlag() { return ProbDistFlag; };

  void setStaticEnvironment(bool _staticEnvironment) { staticEnvironment = _staticEnvironment; };
  bool getStaticEnvironment() { return staticEnvironment; };

  void setOccupancy(const std::vector<RealType> _occupancy)
  {
    occupancy = _occupancy;
  };
  std::vector<RealType> getOccupancy() { return occupancy; };
  unsigned long int getOccupancySize() { return occupancySize; };

  std::vector<RealType> getTransitionProbabilities() { return transitionProbabilities; };

  std::vector<RealType> getSaveOccupancy();

  void resizeOccupancy(unsigned long int size)
  {
    occupancy.insert(occupancy.end(), size, RealType(0));
    occupancySize += size;
  };

  unsigned long int getTime() { return time; };

  void setTime(const unsigned long int _time) { time = _time; };

  std::pair<unsigned long int, unsigned long int>
  getEdges()
  {
    return edges;
  };

  void setEdges(std::pair<unsigned long int,
                          unsigned long int> _edges)
  {
    edges = _edges;
  }

  unsigned long int getMaxIdx() { return edges.second; };
  unsigned long int getMinIdx() { return edges.first; };

  double getSmallCutoff() { return smallCutoff; };
  void setSmallCutoff(const double _smallCutoff)
  {
    smallCutoff = _smallCutoff;
  };

  double getLargeCutoff() { return largeCutoff; };
  void setLargeCutoff(const double _largeCutoff)
  {
    largeCutoff = _largeCutoff;
  };

  // Functions that do things

  void iterateTimestep();

  double findQuantile(const RealType quantile);
  std::vector<double> findQuantiles(std::vector<RealType> quantiles);

  RealType pGreaterThanX(const unsigned long int idx);

  std::pair<std::vector<double>, std::vector<RealType>>
  calcVsAndPb(const unsigned long int num);

  std::pair<std::vector<double>, std::vector<RealType>> VsAndPb(const double v);

  RealType getGumbelVariance(RealType maxParticle);
  std::vector<RealType> getCDF();
  std::pair<std::vector<long int>, std::vector<RealType>> getxvals_and_pdf();
};

#endif /* DIFFUSIONPDF_HPP_ */
