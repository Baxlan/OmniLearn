//metric.cpp

#include "omnilearn/metric.h"



// the inputs are loss, the output is average loss
double omnilearn::averageLoss(Matrix const& loss)
{
    Vector feature(loss.rows());
    for(eigen_size_t i = 0; i < loss.rows(); i++)
    {
        feature[i] = loss.row(i).sum();
    }
    return feature.mean();
}


//first is "accuracy", second is "false prediction"
std::pair<double, double> omnilearn::classificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity)
{
    double validated = 0;
    double fp = 0; //false prediction
    double count = 0; // equals real.size() in case of "one label per data"
                      // but is different in case of multi labeled data

    for(eigen_size_t i = 0; i < real.rows(); i++)
    {
        for(eigen_size_t j = 0; j < real.cols(); j++)
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
    if(validated + fp != 0)
    {
        fp = 100*fp/(validated + fp);
    }
    validated = 100*validated/count;
    return {validated, fp};
}


//first is L1, second is L2, with normalized outputs
std::pair<double, double> omnilearn::regressionMetrics(Matrix real, Matrix predicted, std::vector<std::pair<double, double>> const& normalization)
{
    //"real" are already normalized
    normalize(predicted, normalization);

    Vector mae = Vector::Constant(real.rows(), 0);
    Vector mse = Vector::Constant(real.rows(), 0);

    for(eigen_size_t i = 0; i < real.rows(); i++)
    {
        for(eigen_size_t j = 0; j < real.cols(); j++)
        {
            mae[i] += std::abs(real(i, j) - predicted(i, j));
            mse[i] += std::pow(real(i, j) - predicted(i, j), 2);
        }
    }
    return {mae.mean(), mse.mean()};
}