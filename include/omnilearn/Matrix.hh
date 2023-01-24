// Maxtrix.hh

#ifndef OMNILEARN_MATRIX_HH_
#define OMNILEARN_MATRIX_HH_

#include "disable_eigen_warnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
DISABLE_WARNING_EXTRA
DISABLE_WARNING_DEPRECATED_COPY
DISABLE_WARNING_OLD_STYLE_CAST
DISABLE_WARNING_CONVERSION
#include "eigen/Core"
DISABLE_WARNING_POP

#define eigen_size_t long long



namespace omnilearn
{



using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DiagMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using rowVector = Eigen::Matrix<double, 1, Eigen::Dynamic>;


double dev(Vector const& vec);
double norm(Vector const& vec, double order = 2);
double normInf(Vector const& vec);

Vector stdToEigenVector(std::vector<double> const& vec);
std::vector<double> eigenToStdVector(Vector const& vec);

template<typename Derived>
typename Derived::Scalar median( Eigen::DenseBase<Derived>& d ){
    auto r { d.reshaped() };
    std::sort( r.begin(), r.end() );
    return r.size() % 2 == 0 ?
        r.segment( (r.size()-2)/2, 2 ).mean() :
        r( r.size()/2 );
}


template<typename Derived>
typename Derived::Scalar median( const Eigen::DenseBase<Derived>& d ){
    typename Derived::PlainObject m { d.replicate(1,1) };
    return median(m);
}

} // namespace omnilearn

#endif // OMNILEARN_MATRIX_HH_