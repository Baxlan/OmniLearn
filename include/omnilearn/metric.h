// metric.h

#ifndef OMNILEARN_TEST_H_
#define OMNILEARN_TEST_H_

#include "preprocess.hh"



namespace omnilearn
{



// the inputs are loss, the output is average loss
double averageLoss(Matrix const& loss);
//first is "accuracy", second is "mean positive likelihood", third is "mean negative likelihood", fourth is "mean cohen kappa"
std::array<double, 4> classificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity);
//first is L1 (MAE), second is L2(RMSE), third is "median absolute error", fourth is "mean correlation" , all with normalized outputs
std::array<double, 4> regressionMetrics(Matrix real, Matrix predicted, std::vector<std::pair<double, double>> const& normalization);



} // namespace omnilearn



#endif // OMNILEARN_TEST_H_