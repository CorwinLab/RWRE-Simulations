#include "diffusionCDFBase.hpp"
#include "randomNumGenerator.hpp"

DiffusionCDF::DiffusionCDF(const double _beta, const unsigned long int _tMax) : RandomNumGenerator(_beta)
{
  tMax = _tMax;
}