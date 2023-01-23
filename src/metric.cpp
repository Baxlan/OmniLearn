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


//first is "accuracy", second is "mean positive likelihood", third is "mean negative likelihood", fourth is "mean cohen kappa"
std::array<double, 4> omnilearn::classificationMetrics(Matrix const& real, Matrix const& predicted, double classValidity)
{
    Vector predictionLikelihood = Vector::Constant(real.cols(), 0); //P(score>=threshold | label=1)
    Vector rejectionLikelihood = Vector::Constant(real.cols(), 0);  //P(score<threshold | label=0)
    Vector predictionCount = Vector::Constant(real.cols(), 0);
    Vector rejectionCount = Vector::Constant(real.cols(), 0);

    Vector falsePrediction = Vector::Constant(real.cols(), 0);
    Vector falseRejection = Vector::Constant(real.cols(), 0);

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
                else
                {
                    falseRejection(j)++;
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

    double positiveCohenKappa = 0;
    double negativeCohenKappa = 0;
    // cohen Kappa metric indicate that:
    // 0 = model output is similar than random output
    // 1 = model is perfect
    // inferior to 0 = model is worse than random

    // the next bloc is the real Cohen kappa metric, it only works for monolabel classifiaction
    if(false)
    {
        for(eigen_size_t i = 0; i < real.cols(); i++)
        {
            positiveCohenKappa += (predictionLikelihood(i)+falsePrediction(i)) * predictionCount(i) / std::pow(real.rows(), 2);
            negativeCohenKappa += (rejectionLikelihood(i)+falseRejection(i)) * rejectionCount(i) / std::pow(real.rows(), 2);
        }
        positiveCohenKappa = (accuracy-positiveCohenKappa)/(1-positiveCohenKappa);
        negativeCohenKappa = ((1-accuracy)-negativeCohenKappa)/(1-negativeCohenKappa);
    }
    // the next bloc calculates a mean Cohen kappa over each label, allowing multilabel classification
    // each label is considered binary, thus one cohen kappa is calculated for each label, then the mean is taken.
    else
    {
        for(eigen_size_t i = 0; i < real.cols(); i++)
        {
            double p_po = predictionLikelihood(i)/predictionCount(i);
            double p_pe = (predictionLikelihood(i)+falsePrediction(i)) * predictionCount(i);
            p_pe += (rejectionLikelihood(i)+falseRejection(i)) * rejectionCount(i);
            p_pe /= std::pow(real.rows(), 2);

            double n_po = rejectionLikelihood(i)/rejectionCount(i);
            double n_pe = (rejectionLikelihood(i)+falseRejection(i)) * rejectionCount(i);
            n_pe += (predictionLikelihood(i)+falsePrediction(i)) * predictionCount(i);
            n_pe /= std::pow(real.rows(), 2);

            positiveCohenKappa += (p_po-p_pe)/(1-p_pe);
            negativeCohenKappa += (n_po-n_pe)/(1-n_pe);
        }
        positiveCohenKappa /= static_cast<double>(real.cols());
        negativeCohenKappa /= static_cast<double>(real.cols());
    }

    predictionLikelihood = 100*predictionLikelihood.array()/predictionCount.array(); //P(predicted >= threshold | real=1)
    rejectionLikelihood = 100*rejectionLikelihood.array()/rejectionCount.array();    //P(predicted <  threshold | real=0)

    return {predictionLikelihood.mean(), rejectionLikelihood.mean(), positiveCohenKappa, negativeCohenKappa};
}


//first is L1 (MAE), second is L2(RMSE), third is "median absolute error", fourth is "mean correlation" , all with normalized outputs
std::array<double, 4> omnilearn::regressionMetrics(Matrix real, Matrix predicted, std::vector<std::pair<double, double>> const& normalization)
{
    //"real" are already normalized
    normalize(predicted, normalization);

    //ROW WISE !
    Vector mae = Vector::Constant(real.rows(), 0);
    Vector rmse = Vector::Constant(real.rows(), 0);

    //COLUMN WIZE !
    Vector correlation = Vector::Constant(real.cols(), 0);
    Vector cosine = Vector::Constant(real.cols(), 0);
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
            cosine[j] = real.col(j).dot(predicted.col(j)) / (real.col(j).norm() * predicted.col(j).norm());
        }
    }

    for(eigen_size_t i = 0; i < real.cols(); i++)
    {
        correlation[i] /= std::sqrt((real.col(i).array() - realMeans(i)).square().sum()) * std::sqrt((predicted.col(i).array() - predictedMeans(i)).square().sum());
    }

    mae  =  mae/real.cols();
    rmse = rmse/real.cols();

    return {mae.mean(), std::sqrt(rmse.mean()), correlation.array().abs().mean(), cosine.array().abs().mean()};
}