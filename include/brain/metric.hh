#ifndef BRAIN_TEST_HH_
#define BRAIN_TEST_HH_

#include "Matrix.hh"

namespace brain
{



// the inputs are loss, the output is average loss
double averageLoss(Matrix const& loss)
{
    Vector feature(loss.rows());
    for(size_t i = 0; i < loss.rows(); i++)
    {
        feature[i] = loss.row(i).sum();
    }
    return feature.mean();
}


//first is "accuracy", second is "false prediction"
std::pair<double, double> classificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity)
{
    double validated = 0;
    double fp = 0; //false prediction
    double count = 0; // equals real.size() in case of "one label per data"
                      // but is different in case of multi labeled data

    for(size_t i = 0; i < real.rows(); i++)
    {
        for(size_t j = 0; j < real.cols(); j++)
        {
            if(std::abs(real(i, j) - 1) <= std::numeric_limits<double>::epsilon())
            {
                count++;
                if(predicted(i, j) >= classValidity)
                {
                    validated++;
                }
            }
            else
            {
                if(predicted(i, j) >= classValidity)
                {
                    fp++;
                }
            }
        }
    }
    fp = 100*fp/(validated + fp);
    validated = 100*validated/count;
    return {validated, fp};
}


//first is L1, second is L2, all with normalized outputs
std::pair<double, double> regressionMetrics(Matrix const& real, Matrix const& predicted)
{
    Vector mae = Vector::Constant(real.rows(), 0);
    Vector mse = Vector::Constant(real.rows(), 0);

    //mean absolute error
    for(size_t i = 0; i < real.rows(); i++)
    {
        for(size_t j = 0; j < real.cols(); j++)
        {
            mae[i] += std::abs(real(i, j) - predicted(i, j));
            mse[i] += std::pow(real(i, j) - predicted(i, j), 2);
        }
    }
    return {mae.mean(), mse.mean()};
}



} // namespace brain

#endif // BRAIN_TEST_HH_