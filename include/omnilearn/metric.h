// metric.h

#ifndef OMNILEARN_TEST_H_
#define OMNILEARN_TEST_H_

#include "preprocess.hh"



namespace omnilearn
{



// the inputs are loss, the output is average loss
double averageLoss(Matrix const& loss);
//first is "accuracy", second is "false prediction"
std::array<double, 4> monoClassificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity);
std::array<double, 4> multipleClassificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity);
//first is L1, second is L2, with normalized outputs
std::array<double, 4> regressionMetrics(Matrix real, Matrix predicted, std::vector<std::pair<double, double>> const& normalization);



} // namespace omnilearn



#endif // OMNILEARN_TEST_H_