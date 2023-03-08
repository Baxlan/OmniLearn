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

    double accuracy = 100*predictionLikelihood.array().sum()/predictionCount.sum();
    double cohenKappa = 0;
    // cohen Kappa metric indicates that:
    // 1 => model is perfect
    // 0 => model output is equivalent to random output
    // negative => model is worse than random

    // next loop calculates a mean Cohen kappa over each label, allowing multilabel classification
    // each label is considered binary, thus one cohen kappa is calculated for each label, then the mean is taken.
    for(eigen_size_t i = 0; i < real.cols(); i++)
    {
        double po = (predictionLikelihood(i)+rejectionLikelihood(i))/static_cast<double>(real.rows());
        double pe = (predictionLikelihood(i)+falsePrediction(i)) * predictionCount(i);
        pe += (rejectionLikelihood(i)+falseRejection(i)) * rejectionCount(i);
        pe /= std::pow(real.rows(), 2);

        cohenKappa += (po-pe)/(1-pe);
    }
    cohenKappa /= static_cast<double>(real.cols());

    predictionLikelihood = 100*predictionLikelihood.array()/predictionCount.array(); //P(predicted >= threshold | real=1)
    rejectionLikelihood = 100*rejectionLikelihood.array()/rejectionCount.array();    //P(predicted <  threshold | real=0)

    return {accuracy, predictionLikelihood.mean(), rejectionLikelihood.mean(), cohenKappa};
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

    return {mae.mean(), std::sqrt(rmse.mean()), correlation.mean(), cosine.mean()};
}