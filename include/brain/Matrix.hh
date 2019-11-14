#ifndef BRAIN_MATRIX_HH_
#define BRAIN_MATRIX_HH_


#include "eigen/Core"
#include "eigen/SVD"

namespace brain
{

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> rowVector;


double dev(Vector const& vec)
{
  return std::sqrt((vec.array() - vec.mean()).square().sum()/(vec.size()-1));
}


double norm(Vector const& vec, double order = 2)
{
  double norm = 0;
  for(size_t i = 0; i < vec.size(); i++)
    norm += std::pow(vec[i], order);
  return std::pow(norm, 1/order);
}


double normInf(Vector const& vec)
{
  return vec.maxCoeff();
}


} // namespace brain

#endif // BRAIN_MATRIX_HH_