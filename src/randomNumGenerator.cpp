#include "randomNumGenerator.hpp"

RandomNumGenerator::RandomNumGenerator(const double _beta)
{
  beta = _beta;

  if (_beta != 0) {
    boost::random::beta_distribution<>::param_type params(_beta, _beta);
    betaParams = params;
  }

  std::uniform_real_distribution<>::param_type unifParams(0.0, 1.0);
  dis.param(unifParams);
  gen.seed(rd());
}

double RandomNumGenerator::generateBeta(){
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
        double randomVal = beta_dist(gen, betaParams);
        // This is a hack for now. If beta is too small it may return an NaN value.
        if (isnan(randomVal)){
            randomVal = beta_dist(gen, betaParams);
        }
        return randomVal;
    }
}