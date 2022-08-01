#include <math.h>
#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <math.h>

#include "stat.h"
#include "diffusionTimeCDF.hpp"
#include "diffusionCDFBase.hpp"

DiffusionTimeCDF::DiffusionTimeCDF(const double _beta,
                                   const unsigned long int _tMax)
    : DiffusionCDF(_beta, _tMax)
{
  CDF.resize(tMax + 1);
  CDF[0] = 1;
}

void DiffusionTimeCDF::iterateTimeStep()
{
  std::vector<RealType> CDF_next(tMax + 1);
  for (unsigned long int n = 0; n <= t + 1; n++) {
    if (n == 0) {
      CDF_next[n] = 1; // Need CDF(n=0, t) = 1
    }
    else if (n == t + 1) {
      RealType beta = RealType(generateBeta());
      CDF_next[n] = beta * CDF[n - 1];
    }
    else {
      /* Maybe add this in
      if (CDF[n-1] == CDF[n]){
        continue // or could even break?
      }
      */
      RealType beta = RealType(generateBeta());
      CDF_next[n] = beta * CDF[n - 1] + (1 - beta) * CDF[n];
    }
  }
  CDF = CDF_next;
  t += 1;
}

long int DiffusionTimeCDF::findQuantile(RealType quantile)
{ 
  unsigned long int quantilePosition;
  for (unsigned long int n = t; n >= 0; n--) {
    if (CDF[n] > 1 / quantile) {
      quantilePosition = 2 * n + 2 - t;
      break;
    }
  }
  return quantilePosition;
}

long int DiffusionTimeCDF::findLowerQuantile(RealType quantile) {
  long int quantilePosition=0;
  for (unsigned long int n = 0; n <= t; n++) {
    if (CDF[n] < 1. - 1 / quantile) {
      quantilePosition = 2 * n - t;
      break;
    }
  }
  return quantilePosition;
}


std::vector<long int> DiffusionTimeCDF::findQuantiles(
  std::vector<RealType> quantiles)
{
  // Sort incoming quantiles b/c we need them to be in descending order for
  // algorithm to work
  std::sort(quantiles.begin(), quantiles.end(), std::greater<RealType>());

  // Initialize to have same number of vectors as in Ns
  std::vector<long int> quantilePositions(quantiles.size(), 0);
  unsigned long int quantile_idx = 0;
  for (unsigned long int n = t; n >= 0; n--) {
    while (CDF[n] > 1 / quantiles[quantile_idx]) {
      quantilePositions[quantile_idx] = 2 * n + 2 - t;
      quantile_idx += 1;

      // Break while loop if past last position
      if (quantile_idx == quantiles.size()) {
        break;
      }
    }
    // Also need to break for loop if in last position b/c we are done searching
    if (quantile_idx == quantiles.size()) {
      break;
    }
  }
  return quantilePositions;
}

std::vector<long int> DiffusionTimeCDF::getxvals()
{
  std::vector<long int> xvals(t + 1);
  for (unsigned int n = 0; n < xvals.size(); n++) {
    xvals[n] = 2 * n - t;
  }
  return xvals;
}

RealType DiffusionTimeCDF::getGumbelVariance(RealType nParticles)
{
  nParticles+=1;
  std::vector<RealType> cdf = slice(CDF, 0, t);
  cdf.push_back(0); // Need to add 0 to CDF to make it complete.

  std::vector<long int> xvals = getxvals();
  RealType var = getGumbelVarianceCDF(xvals, cdf, nParticles);
  return var;
}

std::vector<RealType>
DiffusionTimeCDF::getGumbelVariance(std::vector<RealType> nParticles)
{
  std::vector<RealType> cdf = slice(CDF, 0, t);
  cdf.push_back(0); // Need to add 0 to CDF to make it complete.
  std::vector<long int> xvals = getxvals();
  return getGumbelVarianceCDF(xvals, cdf, nParticles);
}

std::pair<RealType, float> DiffusionTimeCDF::getProbandV(RealType quantile)
{
  unsigned long int quantilePosition;
  RealType prob;
  for (unsigned long int n = t; n >= 0; n--) {
    if (CDF[n] > 1 / quantile) {
      quantilePosition = 2 * n - t;
      prob = CDF[n];
      break;
    }
  }
  float v = (float)quantilePosition / (float)t;
  return std::make_pair(prob, v);
}

std::vector<RealType> DiffusionTimeCDF::getSaveCDF()
{
  return slice(CDF, 0, t);
}
