// Matrix.cpp

#include "omnilearn/Matrix.hh"

#define eigen_size_t long long



double omnilearn::dev(Vector const& vec)
{
  return std::sqrt((vec.array() - vec.mean()).square().sum()/(static_cast<double>(vec.size())-1));
}


double omnilearn::norm(Vector const& vec, double order)
{
  double norm = 0;
  for(eigen_size_t i = 0; i < vec.size(); i++)
    norm += std::pow(vec[i], order);
  return std::pow(norm, 1/order);
}


double omnilearn::normInf(Vector const& vec)
{
  return vec.maxCoeff();
}


omnilearn::Vector  omnilearn::stdToEigenVector(std::vector<double> const& vec)
{
  Vector vec2(vec.size());
  std::transform(vec.begin(), vec.end(), vec2.begin(), [](double a) -> double {return a;});
}


std::vector<double>  omnilearn::eigenToStdVectorr(Vector const& vec)
{
  std::vector<double> vec2(vec.size());
  std::transform(vec.begin(), vec.end(), vec2.begin(), [](double a) -> double {return a;});
}