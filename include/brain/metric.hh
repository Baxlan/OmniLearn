#ifndef BRAIN_TEST_HH_
#define BRAIN_TEST_HH_

#include "Matrix.hh"

namespace brain
{



// the inputs are loss, the output is average loss
double averageLoss(Matrix const& loss)
{
    Vector feature(loss.lines(), 0);
    for(unsigned i = 0; i < loss.lines(); i++)
    {
        feature[i] = loss[i].sum();
    }
    return feature.mean().first;
}


//first is "accuracy", second is "false prediction"
std::pair<double, double> accuracy(Matrix const& real, Matrix const& predicted, double classValidity)
{
    double validated = 0;
    double fp = 0; //false prediction
    double count = 0; // equals real.size() in case of "one label per data"
                      // but is different in case of multi labeled data

    for(unsigned i = 0; i < real.lines(); i++)
    {
        for(unsigned j = 0; j < real.columns(); j++)
        {
            if(std::abs(real[i][j] - 1) <= std::numeric_limits<double>::epsilon())
            {
                count++;
                if(predicted[i][j] >= classValidity)
                {
                    validated++;
                }
            }
            else
            {
                if(predicted[i][j] >= classValidity)
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


//first is "accuracy", second is "false prediction"
std::pair<Vector, Vector> accuracyPerOutput(Matrix const& real, Matrix const& predicted, double classValidity)
{
    Vector validated(real[0].size(), 0);
    Vector fp(real[0].size(), 0); //false prediction
    std::vector<unsigned> count(real[0].size(), 0);

    for(unsigned i = 0; i < real.lines(); i++)
    {
        for(unsigned j = 0; j < real.columns(); j++)
        {
            if(std::abs(real[i][j] - 1) <= std::numeric_limits<double>::epsilon())
            {
                count[j]++;
                if(predicted[i][j] >= classValidity)
                {
                    validated[j]++;
                }
            }
            else
            {
                if(predicted[i][j] >= classValidity)
                {
                    fp[j]++;
                }
            }
        }
    }
    for(unsigned i = 0; i < real[0].size(); i++)
    {
        validated[i] = 100*validated[i]/count[i];
        fp[i] = 100*fp[i]/(validated[i]+fp[i]);
    }
    return {validated, fp};
}



//first is mae with normalized outputs, second is with unormalized ones
std::pair<double, double> L1Metric(Matrix const& real, Matrix const& predicted, std::vector<std::pair<double, double>> const& mM)
{
    Vector sums(real.lines(), 0);
    Vector unormalizedSums(real.lines(), 0);

    //mean absolute error
    for(unsigned i = 0; i < real.lines(); i++)
    {
        for(unsigned j = 0; j < real.columns(); j++)
        {
            sums[i] += std::abs(real[i][j] - predicted[i][j]);
            unormalizedSums[i] += std::abs(real[i][j] - predicted[i][j]) * std::abs(mM[j].second - mM[j].first);
        }
    }

    return {sums.mean().first, unormalizedSums.mean().first};
}


//first is mse with normalized outputs, second is with unormalized ones
std::pair<double, double> L2Metric(Matrix const& real, Matrix const& predicted, std::vector<std::pair<double, double>> const& mM)
{
    Vector sums(real.lines(), 0);
    Vector unormalizedSums(real.lines(), 0);

    //mean square error
    for(unsigned i = 0; i < real.lines(); i++)
    {
        for(unsigned j = 0; j < real.columns(); j++)
        {
            sums[i] += std::pow(real[i][j] - predicted[i][j], 2);
            unormalizedSums[i] += std::pow(real[i][j] - predicted[i][j], 2) * std::pow(mM[j].second - mM[j].first, 2);
        }
    }

    return {sums.mean().first, unormalizedSums.mean().first};
}



} // namespace brain

#endif // BRAIN_TEST_HH_