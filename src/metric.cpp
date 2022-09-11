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
std::array<double, 4> omnilearn::monoClassificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity)
{
    double validated = 0;
    double fp = 0; //false positive prediction
    double fn = 0; //false negative prediction

    for(eigen_size_t i = 0; i < real.rows(); i++)
    {
        for(eigen_size_t j = 0; j < real.cols(); j++)
        {
            if(std::abs(real(i, j) - 1) <= std::numeric_limits<double>::epsilon())
            {
                if(predicted(i, j) >= classValidity)
                {
                    validated++;
                }
                else
                {
                    fn++;
                }
                break;
            }
            else
            {
                if(predicted(i, j) >= classValidity)
                {
                    fp++;
                    break;
                }
            }
        }
    }
    // because it is mono labeled date, true positive likelihood = true negative likelihood
    fp = 100*fp/(validated + fp);
    fn = 100*fn/(validated + fn);
    validated = 100*validated/static_cast<double>(real.rows());

    return {validated, fp, validated, fn};
}


//first is "accuracy", second is "false prediction"
std::array<double, 4> omnilearn::multipleClassificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity)
{
    double validated = 0;
    double rejected = 0;
    double actualPositives = 0;
    double actualNegatves = 0;
    double fp = 0; //false positive prediction
    double fn = 0; //false negative prediction

    for(eigen_size_t i = 0; i < real.rows(); i++)
    {
        for(eigen_size_t j = 0; j < real.cols(); j++)
        {
            if(std::abs(real(i, j) - 1) <= std::numeric_limits<double>::epsilon())
            {
                actualPositives++;
                if(predicted(i, j) >= classValidity)
                {
                    validated++;
                }
                else
                {
                    fn++;
                }
            }
            else
            {
                actualNegatves++;
                if(predicted(i, j) >= classValidity)
                {
                    fp++;
                }
                else
                {
                    rejected++;
                }
            }
        }
    }
    fp = 100*fp/(validated + fp);
    fn = 100*fn/(rejected + fn);
    validated = 100*validated/actualPositives;
    rejected = 100*rejected/actualNegatves;

    return {validated, fp, rejected, fn};
}


//first is L1 (MAE), second is L2(RMSE), with normalized outputs
std::array<double, 4> omnilearn::regressionMetrics(Matrix real, Matrix predicted, std::vector<std::pair<double, double>> const& normalization, size_t inputVariables)
{
    //"real" are already normalized
    normalize(predicted, normalization);

    Vector mae = Vector::Constant(real.rows(), 0);
    Vector rmse = Vector::Constant(real.rows(), 0);
    Vector adjR2 = Vector::Constant(real.rows(), 0);
    Vector correlation = Vector::Constant(real.rows(), 0);

    Vector realMeans = Vector::Constant(real.rows(), 0);
    Vector predictedMeans = Vector::Constant(real.rows(), 0);

    for(eigen_size_t i = 0; i < real.cols(); i++)
    {
        realMeans[i] = real.col(i).mean();
        predictedMeans[i] = predicted.col(i).mean();
    }

    for(eigen_size_t i = 0; i < real.rows(); i++)
    {
        for(eigen_size_t j = 0; j < real.cols(); j++)
        {
            mae[i] += std::abs(real(i, j) - predicted(i, j));
            rmse[i] += std::pow(real(i, j) - predicted(i, j), 2);
            adjR2[i] += std::pow(realMeans(i, j) - predicted(i, j), 2);
            correlation[i] += (predicted(i, j)-predictedMeans(i))*(real(i, j)-realMeans(i));
        }
    }

    for(eigen_size_t i = 0; i < real.cols(); i++)
    {
        adjR2[i] = 1 - (rmse[i] / adjR2[i]);
        adjR2[i] = 1 - ((1-adjR2[i]) * static_cast<double>((real.rows()-1)) / static_cast<double>(real.rows()-inputVariables-1));

        correlation[i] /= std::sqrt((real.array() - realMeans(i)).square().sum()) * std::sqrt((predicted.array() - predictedMeans(i)).square().sum());
    }

    return {mae.mean(), std::sqrt(rmse.mean()), adjR2.mean(), correlation.mean()};
}