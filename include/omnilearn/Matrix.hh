// Maxtrix.hh

#ifndef OMNILEARN_MATRIX_HH_
#define OMNILEARN_MATRIX_HH_

#include "disable_eigen_warnings.hh"

DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
DISABLE_WARNING_EXTRA
DISABLE_WARNING_OLD_STYLE_CAST
DISABLE_WARNING_CONVERSION
#include "eigen/Core"
DISABLE_WARNING_POP

#define eigen_size_t long long



namespace omnilearn
{



using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using rowVector = Eigen::Matrix<double, 1, Eigen::Dynamic>;


double dev(Vector const& vec);
double norm(Vector const& vec, double order = 2);
double normInf(Vector const& vec);


} // namespace omnilearn

#endif // OMNILEARN_MATRIX_HH_