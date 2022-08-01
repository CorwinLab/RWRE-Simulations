#include <cmath>
#include <vector>
#include <iostream>
#include <utility>
#include <string>

template<typename T>
std::vector<T> slice(std::vector<T> &v, const unsigned long int m, const unsigned long int n){
  std::vector<T> vec(n - m + 1);
  std::copy(v.begin() + m, v.begin() + n + 1, vec.begin());
  return vec;
}

template<class T, class N>
void checkVectorLengths(std::vector<T> vec1, std::vector<N> vec2){
  if (vec1.size() != vec2.size()){
    std::string error = "Vectors must have same length but sizes: " + std::to_string(vec1.size()) + ", " + std::to_string(vec2.size());
    throw std::runtime_error(error);
  }
}

template <class RealType, class x_numeric>
RealType calculateMeanFromPDF(std::vector<x_numeric> xvals, std::vector<RealType> PDF){
  RealType mean = 0;
  for (unsigned long int i=0; i < PDF.size(); i++){
    mean += xvals[i] * PDF[i];
  }
  return mean;
}

template <class RealType, class x_numeric>
RealType calculateVarianceFromPDF(std::vector<x_numeric> xvals, std::vector<RealType> PDF){
  checkVectorLengths(xvals, PDF);
  RealType mean = calculateMeanFromPDF(xvals, PDF);
  RealType var = 0;
  for (unsigned long int i=0; i < PDF.size(); i++){
    var += PDF[i] * pow(xvals[i] - mean, 2);
  }
  return var;
}

template <class RealType, class x_numeric>
RealType calculateMeanFromCDF(std::vector<x_numeric> xvals, std::vector<RealType> cdf)
{
  RealType mean = 0;
  int H;
  for (unsigned long int i=0; i < cdf.size(); i++){
    if (xvals.at(i) <= 0){
      H = 0;
    }
    else{
      H = 1;
    }
    mean += H - cdf[i];
  }
  return mean;
}

template <class RealType>
std::vector<RealType> getDiscretePDF(std::vector<RealType> comp_cdf, RealType nParticles){
  std::vector<RealType> pdf(comp_cdf.size()-1);
  RealType cdf_current, cdf_prev;
  for (unsigned long int i=1; i < comp_cdf.size(); i++){
    cdf_current = exp(-comp_cdf[i] * nParticles);
    cdf_prev = exp(-comp_cdf[i-1] * nParticles);
    pdf[i-1] = cdf_current - cdf_prev;
  }
  return pdf;
}

template <class RealType>
std::vector<std::vector<RealType> > getDiscretePDF(std::vector<RealType> comp_cdf, std::vector<RealType> nParticles){
  // Need to construct a 2D array where each element is a PDF for a different particle.
  std::vector<std::vector<RealType> > pdf(nParticles.size(), std::vector<RealType>(comp_cdf.size()-1));

  RealType cdf_current, cdf_prev;
  for (unsigned long int i=1; i < comp_cdf.size(); i++){
    for (unsigned int j=0; j < nParticles.size(); j++){
      cdf_current = exp(-comp_cdf[i] * nParticles[j]);
      cdf_prev = exp(-comp_cdf[i-1] * nParticles[j]);
      pdf[j][i-1] = cdf_current - cdf_prev;
    }
  }
  return pdf;
}

template <class RealType>
std::vector<RealType> pdf_to_comp_cdf(std::vector<RealType> pdf, RealType norm){
  std::vector<RealType> comp_cdf(pdf.size()+1);
  comp_cdf[0] = 1;
  RealType sum = 0;
  for (unsigned int i = 0; i < pdf.size(); i++){
    sum += pdf[i];
    comp_cdf[i+1] = 1.0 - sum / norm;
  }
  return comp_cdf;
}

// Note: comp_cdf should have size xvals.size()+1 since running getDiscretePDF returns
// a vector of size comp_cdf.size() - 1
template <class RealType, class x_numeric>
RealType getGumbelVarianceCDF(std::vector<x_numeric> xvals, std::vector<RealType> comp_cdf, RealType nParticles){
  std::vector<RealType> discrete_pdf = getDiscretePDF(comp_cdf, nParticles);
  return calculateVarianceFromPDF(xvals, discrete_pdf);
}

template <class RealType, class x_numeric>
std::vector<RealType> getGumbelVarianceCDF(std::vector<x_numeric> xvals, std::vector<RealType> comp_cdf, std::vector<RealType> nParticles){
  std::vector<std::vector<RealType> > discrete_pdf = getDiscretePDF(comp_cdf, nParticles);
  std::vector<RealType> gumbelVariance(nParticles.size());
  for (unsigned int i=0; i < nParticles.size(); i++){
    gumbelVariance[i] = calculateVarianceFromPDF(xvals, discrete_pdf[i]);
  }
  return gumbelVariance;
}

template<class RealType, class x_numeric>
RealType getGumbelVariancePDF(std::vector<x_numeric> xvals, std::vector<RealType> pdf, RealType nParticles, RealType norm){
  std::vector<RealType> comp_cdf = pdf_to_comp_cdf(pdf, norm);
  return getGumbelVarianceCDF(xvals, comp_cdf, nParticles);
}
