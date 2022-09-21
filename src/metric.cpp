//metric.cpp

#include "omnilearn/metric.h"

#include <iostream>

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
    Vector predictionLikelihood = Vector::Constant(real.cols(), 0); //P(score>=threshold | label=1)
    Vector rejectionLikelihood = Vector::Constant(real.cols(), 0);  //P(score<threshold | label=0)
    Vector predictionCount = Vector::Constant(real.cols(), 0);
    Vector rejectionCount = Vector::Constant(real.cols(), 0);

    Vector falsePrediction = Vector::Constant(real.cols(), 0);

    for(eigen_size_t i = 0; i < real.rows(); i++)
    {
        for(eigen_size_t j = 0; j < predicted.cols(); j++)
        {
            if(std::abs(real(i, j) - 1) <= std::numeric_limits<double>::epsilon())
            {
                predictionCount(j)++;
                if(predicted(i, j) >= classValidity)
                {
                    predictionLikelihood(j)++;
                }
            }
            else
            {
                rejectionCount(j)++;
                if(predicted(i, j) >= classValidity)
                {
                    falsePrediction(j)++;
                }
                else
                {
                    rejectionLikelihood(j)++;
                }
            }
        }
    }
    double accuracy = predictionLikelihood.array().sum()/static_cast<double>(real.rows());

    // cohen Kappa metric indicate that:
    // 0 = model output is similar than random output
    // 1 = model is perfect
    // inferior to 0 = model is worse than random
    double cohenKappa = 0;
    for(eigen_size_t i = 0; i < real.cols(); i++)
    {
        cohenKappa += (predictionLikelihood(i)+falsePrediction(i)) * predictionCount(i) / std::pow(real.rows(), 2);
    }
    cohenKappa = (accuracy-cohenKappa)/(1-cohenKappa);

    predictionLikelihood = 100*predictionLikelihood.array()/predictionCount.array(); //P(score>=threshold | label=1)
    rejectionLikelihood = 100*rejectionLikelihood.array()/rejectionCount.array();    //P(score<threshold | label=0)

    return {100*accuracy, predictionLikelihood.mean(), rejectionLikelihood.mean(), cohenKappa};
}


//first is "accuracy", second is "false prediction"
std::array<double, 4> omnilearn::multipleClassificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity)
{
     Vector predictionLikelihood = Vector::Constant(real.cols(), 0); //P(score>=threshold | label=1)
    Vector rejectionLikelihood = Vector::Constant(real.cols(), 0);  //P(score<threshold | label=0)
    Vector predictionCount = Vector::Constant(real.cols(), 0);
    Vector rejectionCount = Vector::Constant(real.cols(), 0);

    Vector falsePrediction = Vector::Constant(real.cols(), 0);

    for(eigen_size_t i = 0; i < real.rows(); i++)
    {
        for(eigen_size_t j = 0; j < predicted.cols(); j++)
        {
            if(std::abs(real(i, j) - 1) <= std::numeric_limits<double>::epsilon())
            {
                predictionCount(j)++;
                if(predicted(i, j) >= classValidity)
                {
                    predictionLikelihood(j)++;
                }
            }
            else
            {
                rejectionCount(j)++;
                if(predicted(i, j) >= classValidity)
                {
                    falsePrediction(j)++;
                }
                else
                {
                    rejectionLikelihood(j)++;
                }
            }
        }
    }
    double accuracy = predictionLikelihood.array().sum()/predictionCount.sum();

    // cohen Kappa metric is invalid for multilabel data
    // here is the krippendorff alpha metric, generalizing cohen kappa because it handle
    // multilabel data, incomplete data (but in our case, a confusion matrix is full), and can weight each label (useless here ?)

    // interpretation:
    // 0 = model output is similar than random output
    // 1 = model is perfect
    // inferior to 0 = model is worse than random

    // https://www.surgehq.ai/blog/inter-rater-reliability-metrics-an-introduction-to-krippendorffs-alpha

    double cohenKappa = 0;
    for(eigen_size_t i = 0; i < real.cols(); i++)
    {
        cohenKappa += (predictionLikelihood(i)+falsePrediction(i)) * predictionCount(i) / std::pow(real.rows(), 2);
    }
    cohenKappa = (accuracy-cohenKappa)/(1-cohenKappa);

    predictionLikelihood = 100*predictionLikelihood.array()/predictionCount.array(); //P(score>=threshold | label=1)
    rejectionLikelihood = 100*rejectionLikelihood.array()/rejectionCount.array();    //P(score<threshold | label=0)

    return {100*accuracy, predictionLikelihood.mean(), rejectionLikelihood.mean(), cohenKappa};
}


//first is L1 (MAE), second is L2(RMSE), with normalized outputs
std::array<double, 4> omnilearn::regressionMetrics(Matrix real, Matrix predicted, std::vector<std::pair<double, double>> const& normalization)
{
    //"real" are already normalized
    normalize(predicted, normalization);

    //ROW WISE !
    Vector mae = Vector::Constant(real.rows(), 0);
    Vector rmse = Vector::Constant(real.rows(), 0);

    //COLUMN WIZE !
    Vector correlation = Vector::Constant(real.cols(), 0);

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
            //ROW WISE !
            mae[i] += std::abs(real(i, j) - predicted(i, j));
            rmse[i] += std::pow(real(i, j) - predicted(i, j), 2);

            //COLUMN WIZE !
            correlation[j] += (predicted(i, j)-predictedMeans(j))*(real(i, j)-realMeans(j));
        }
    }

    for(eigen_size_t i = 0; i < real.cols(); i++)
    {
        correlation[i] /= std::sqrt((real.col(i).array() - realMeans(i)).square().sum()) * std::sqrt((predicted.col(i).array() - predictedMeans(i)).square().sum());
    }

    mae = mae.array()/real.cols();

    return {mae.mean(), std::sqrt((rmse.array()/real.cols()).mean()), median(mae), correlation.array().abs().mean()};
}