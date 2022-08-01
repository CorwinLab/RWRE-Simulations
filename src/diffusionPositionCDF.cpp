#include <math.h>
#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <math.h>

#include "stat.h"
#include "diffusionPositionCDF.hpp"
#include "diffusionCDFBase.hpp"

DiffusionPositionCDF::DiffusionPositionCDF(const double _beta,
                                           const unsigned long int _tMax,
                                           std::vector<RealType> _quantiles)
    : DiffusionCDF(_beta, _tMax)
{
  // Initialize such that CDF(n=0, t) = 1
  CDF.resize(tMax + 1, 1);
  quantiles = _quantiles;

  // Create a measurment array such that each column is a different quantile
  // measurement and each row is a differeent time.
  quantilePositions.resize(quantiles.size());
  for (unsigned long int i = 0; i < quantilePositions.size(); i++) {
    quantilePositions[i].resize(tMax + 1, 1);
    // At t=0, all the quantiles will be 2 (2*n + 2 - t for n=t=0) - I think
    // this is mainly for testing.
    quantilePositions[i][0] = 2;
  }
}

void DiffusionPositionCDF::stepPosition()
{
  // Initialize all values to 0 since below CDF(n, n) is 0
  std::vector<RealType> CDF_next(tMax + 1, 0);

  for (unsigned long int t = position + 1; t < tMax + 1; t++) {
    RealType beta = RealType(generateBeta());
    if (t == position + 1) {
      CDF_next[t] = beta * CDF[t - 1];
    }
    else {
      CDF_next[t] = beta * CDF[t - 1] + (1 - beta) * CDF_next[t - 1];
    }

    // Now want to loop through every quantile vector and update positions of
    // each quantile at the current time.
    for (unsigned long int i = 0; i < quantiles.size(); i++) {
      if (CDF_next[t] > 1 / quantiles[i]) {
        // quantilePositions[i][t] is the ith quantile in the list at time t
        // Note: position will always be greater so don't need to check this.
        quantilePositions[i][t] = 2 * (position + 1) + 2 - t;
      }
    }
  }
  CDF = CDF_next;
  position += 1;
}
