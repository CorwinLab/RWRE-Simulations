#ifndef DIFFUSIONCDFBASE_HPP_
#define DIFFUSIONCDFBASE_HPP_

#include <vector>
#include <boost/multiprecision/float128.hpp>
#include "randomNumGenerator.hpp"

typedef boost::multiprecision::float128 RealType;

class DiffusionCDF : public RandomNumGenerator {
protected:
  std::vector<RealType> CDF;
  unsigned long int tMax;

public:
  DiffusionCDF(const double _beta, const unsigned long int _tMax);
  ~DiffusionCDF(){};

  std::vector<RealType> getCDF() { return CDF; };
  void setCDF(std::vector<RealType> _CDF) { CDF = _CDF; };

  unsigned long int gettMax() { return tMax; };
  void settMax(unsigned long int _tMax) { tMax = _tMax; };

};

#endif /* DIFFUSIONCDFBASE_HPP_ */
