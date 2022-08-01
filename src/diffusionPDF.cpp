#include "diffusionPDF.hpp"
#include "randomNumGenerator.hpp"
#include "stat.h"

#include <math.h>
#include <algorithm>
#include <boost/multiprecision/float128.hpp>
#include <cmath>
#include <limits>
#include <math.h>

using RealType = boost::multiprecision::float128;
static_assert(sizeof(RealType) == 16, "Bad size");

// Constuctor
DiffusionPDF::DiffusionPDF(const RealType _nParticles,
                           const double _beta,
                           const unsigned long int _occupancySize,
                           const bool _ProbDistFlag,
                           const bool _staticEnvironment) 
      : RandomNumGenerator(_beta), nParticles(_nParticles), occupancySize(_occupancySize),
      ProbDistFlag(_ProbDistFlag), staticEnvironment(_staticEnvironment)
{
  if (isnan(nParticles) || isinf(nParticles)) {
    throw std::runtime_error("Number of particles initialized to NaN");
  }
  edges.first = 0, edges.second = 0;

  occupancy.resize(_occupancySize + 1);
  occupancy[0] = nParticles;

  transitionProbabilities.resize(2*_occupancySize +1, -1);
  time = 0;
}

std::vector<RealType> DiffusionPDF::getSaveOccupancy()
{
  return slice(occupancy, edges.first, edges.second);
}

RealType DiffusionPDF::toNextSite(RealType currentSite, RealType bias)
{
  // If generating the probability distribution just default to the
  // number of particles * bias
  if (ProbDistFlag) {
    return (currentSite * bias);
  }

  // The boost binomial can sometimes return negative numbers (or inf) for
  // large or small biases. So we default to number of particles * bias
  if (bias >= 0.99999 || bias <= 0.000001) {
    return (currentSite * bias);
  }

  // For smallCutoff need to downcast currentSite to double. And then cast
  // answer to RealType.
  if (currentSite < smallCutoff) {

    return RealType(binomial(gen,
                             boost::random::binomial_distribution<>::param_type(
                                 double(currentSite), double(bias))));
  }

  else if (currentSite > largeCutoff) {
    return (currentSite * bias);
  }
  // If less than largeCutoff use sqrt(N * p * (1-p)) * randn(-1, 1)
  else {

    RealType mediumVariance = sqrt(currentSite * bias * (1 - bias));
    return currentSite * bias + mediumVariance * (2 * RealType(dis(gen)) - 1);
  }
}

double DiffusionPDF::generateBeta()
{
  // If beta = 0 return either 0 or 1
  if (beta == 0.0) {
    return round(dis(gen));
  }
  // If beta = 1 use random uniform distribution
  else if (beta == 1.0) {
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

void DiffusionPDF::iterateTimestep()
{
  unsigned long int prevMinIndex = edges.first;
  unsigned long int prevMaxIndex = edges.second;
  if (prevMinIndex > prevMaxIndex) {
    throw std::runtime_error(
        "Minimum edge must be greater than maximum edge: (" +
        std::to_string(prevMinIndex) + ", " + std::to_string(prevMaxIndex) +
        ")");
  }

  // If iterating over the whole array extend the occupancy.
  if ((prevMaxIndex + 1) == occupancy.size()) {
    occupancy.push_back(0);
    std::cout << "Warning: pushing back occupancy size. If this happens a lot "
                 "it may effect performance."
              << std::endl;
  }

  RealType fromLastSite = 0;
  RealType toNextSite = 0;
  unsigned long int minEdge = 0;
  unsigned long int maxEdge = 0;
  bool firstNonzero = true;

  for (auto i = prevMinIndex; i < prevMaxIndex + 2; i++) {
    RealType *occ = &occupancy.at(i);

    RealType bias = 0;
    if (*occ != 0) {
      if (!staticEnvironment){
        bias = RealType(DiffusionPDF::generateBeta());
      }
      else {
        unsigned int index = 2 * i - time + transitionProbabilities.size()/2;
        if (transitionProbabilities[index] == -1){
          // The issue is here. Since the actual values of the index don't match 
          // the x-values just doing [i] isn't correct
          transitionProbabilities[index] = RealType(DiffusionPDF::generateBeta());  
        }
        bias = transitionProbabilities[index];
      }
      
      toNextSite = DiffusionPDF::toNextSite(*occ, bias);
      if (!ProbDistFlag) {
        toNextSite = round(toNextSite);
      }
    }
    else {
      toNextSite = 0;
    }

    RealType prevOcc = *occ; // For error checking below
    *occ += fromLastSite - toNextSite;
    fromLastSite = toNextSite;

    if (*occ != 0) {
      maxEdge = i;
      if (firstNonzero) {
        minEdge = i;
        firstNonzero = false;
      }
    }

    if (toNextSite < 0 || toNextSite > prevOcc || bias < 0.0 || bias > 1.0 ||
        *occ < 0 || *occ > nParticles || isnan(*occ)) {
      std::cout << "Time:" << time << "\n";
      std::cout << "Occupancy: " << *occ << "\n";
      std::cout << "Next site: " << toNextSite << "\n";
      std::cout << "Bias: " << bias << std::endl;
      throw std::runtime_error("One or more variables out of bounds: ");
    }
  }

  edges.first = minEdge;
  edges.second = maxEdge;
  time += 1;
}

/*
The algorithm just looks at the sum of the occupancy up to some position maxIdx.
The position of the Nth quartile then happens when the condition
sum > nParticles / quantile.
*/
double DiffusionPDF::findQuantile(const RealType quantile)
{
  unsigned long int maxIdx = edges.second;
  double centerIdx = time * 0.5;

  double dist = maxIdx - centerIdx;
  RealType sum = occupancy.at(maxIdx);
  while (sum < nParticles / quantile) {
    maxIdx -= 1;
    dist -= 1;
    sum += occupancy.at(maxIdx);
  }
  return dist;
}

/*
The algorithm just looks at the sum of the occupancy up to some position maxIdx.
The position of the Nth quartile then happens when the condition
sum > nParticles / quantile.
*/
std::vector<double> DiffusionPDF::findQuantiles(std::vector<RealType> quantiles)
{

  // Need Quantiles in descending order for algorithm to work correctly
  std::sort(quantiles.begin(), quantiles.end(), std::greater<RealType>());

  std::vector<double> dists(quantiles.size());

  unsigned long int maxIdx = edges.second;
  double centerIdx = time * 0.5;
  double dist = maxIdx - centerIdx;
  RealType sum = occupancy.at(maxIdx);

  unsigned long int quantiles_idx = 0;
  while (quantiles_idx < quantiles.size()) {
    while (sum < nParticles / quantiles[quantiles_idx]) {
      maxIdx -= 1;
      dist -= 1;
      sum += occupancy.at(maxIdx);
    }
    dists[quantiles_idx] = dist;
    quantiles_idx += 1;
  }
  return dists;
}

RealType DiffusionPDF::pGreaterThanX(const unsigned long int idx)
{
  RealType Nabove = 0.0;
  for (unsigned long int j = idx; j <= time; j++) {
    Nabove += occupancy.at(j);
  }
  return Nabove / nParticles;
}

std::pair<std::vector<double>, std::vector<RealType>>
DiffusionPDF::calcVsAndPb(const unsigned long int num)
{
  std::vector<double> vs;
  std::vector<RealType> Pbs;
  unsigned long int maxIdx = edges.second;
  RealType Nabove = 0.0;
  for (unsigned long int i = maxIdx; i > (maxIdx - num); i--) {
    Nabove += occupancy.at(i);
    RealType probAbove = Nabove / nParticles;
    double v = (2. * i - time) / time;
    vs.push_back(v);
    Pbs.push_back(probAbove);
  }
  std::pair<std::vector<double>, std::vector<RealType>> returnTuple(vs, Pbs);
  return returnTuple;
}

std::pair<std::vector<double>, std::vector<RealType>>
DiffusionPDF::VsAndPb(const double v)
{
  std::vector<double> vs;
  std::vector<RealType> Pbs;
  unsigned long int idx = edges.second;
  RealType Nabove = 0.0;
  double currentV = (2. * idx - time) / time;
  while (currentV >= v) {
    Nabove += occupancy.at(idx);
    RealType probAbove = Nabove / nParticles;
    vs.push_back(currentV);
    Pbs.push_back(probAbove);

    idx -= 1;
    currentV = (2. * idx - time) / time;
  }
  std::pair<std::vector<double>, std::vector<RealType>> returnTuple(vs, Pbs);
  return returnTuple;
}

std::pair<std::vector<long int>, std::vector<RealType>>
DiffusionPDF::getxvals_and_pdf()
{
  unsigned long int minIdx = edges.first;
  unsigned long int maxIdx = edges.second;

  if (minIdx == 0) {
    minIdx += 1;
  }

  if (maxIdx == occupancy.size() - 1) {
    maxIdx -= 1;
  }

  std::vector<long int> xvals(maxIdx - minIdx + 2);
  std::vector<RealType> pdf(maxIdx - minIdx + 2);

  for (unsigned long int i = minIdx - 1; i <= maxIdx; i++) {
    xvals.at(i - minIdx + 1) = 2 * i - time;
    pdf.at(i - minIdx + 1) = occupancy.at(i);
  }
  return std::make_pair(xvals, pdf);
}

std::vector<RealType> DiffusionPDF::getCDF()
{
  std::pair<std::vector<long int>, std::vector<RealType>> pair =
      getxvals_and_pdf();
  std::vector<RealType> pdf = pair.second;
  return pdf_to_comp_cdf(pdf, nParticles);
}

RealType DiffusionPDF::getGumbelVariance(RealType maxParticle)
{
  std::pair<std::vector<long int>, std::vector<RealType>> pair =
      getxvals_and_pdf();
  std::vector<long int> xvals = pair.first;
  std::vector<RealType> pdf = pair.second;
  return getGumbelVariancePDF(xvals, pdf, maxParticle, nParticles);
}

/*
Trying to hackily get the PDF for the Einstein random walk. Doesn't look like
it works very well b/c there are a lot of zeros in the PDF and it only sums
to 0.5
*/
RealType getEinsteinPDF(unsigned long int n, unsigned long int k)
{
  if (k == 0) {
    return RealType(1 / pow(2, n));
  }
  RealType product = 1;
  unsigned long int multiples = floor(n / k);
  unsigned long int remainder = n % k;
  for (RealType i = 1; i <= k; i++) {
    product *= (n + 1 - i) / i / pow(2, multiples);
  }
  product /= pow(2, remainder);
  return product;
}

std::vector<RealType> getWholeEinsteinPDF(unsigned long int n)
{

  std::vector<RealType> pdf(n + 1);
  for (unsigned long int i = 0; i <= n; i++) {
    pdf.at(i) = getEinsteinPDF(n, i);
  }
  return pdf;
}
