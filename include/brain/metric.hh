#ifndef BRAIN_TEST_HH_
#define BRAIN_TEST_HH_

#include "vectorial.hh"

namespace brain
{



// the inputs are loss, the output is average loss
double averageLoss(Matrix const& loss)
{
    std::vector<double> feature(loss.size(), 0);
    for(unsigned i = 0; i < loss.size(); i++)
    {
        feature[i] = sum(loss[i]);
    }
    return average(feature).first;
}


//first is "accuracy", second is "false prediction"
std::pair<double, double> accuracy(Matrix const& real, Matrix const& predicted, double classValidity)
{
    double validated = 0;
    double fp = 0; //false prediction
    double count = 0; // equals real.size() in case of "one label per data"
                      // but is different is case of multi labaled data

    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
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
    validated = 100*validated/count;
    fp = 100*fp/(validated + fp);
    return {validated, fp};
}


//first is "accuracy", second is "false prediction"
std::pair<std::vector<double>, std::vector<double>> accuracyPerOutput(Matrix const& real, Matrix const& predicted, double classValidity)
{
    std::vector<double> validated(real[0].size(), 0);
    std::vector<double> fp(real[0].size(), 0); //false prediction
    std::vector<unsigned> count(real[0].size(), 0);

    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
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
    std::vector<double> sums(real.size(), 0);
    std::vector<double> unormalizedSums(real.size(), 0);

    //mean absolute error
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            sums[i] += std::abs(real[i][j] - predicted[i][j]);
            unormalizedSums[i] += std::abs(real[i][j] - predicted[i][j]) * std::abs(mM[j].second - mM[j].first);
        }
    }

    return {average(sums).first, average(unormalizedSums).first};
}


//first is mae (of normalized outputs), second is std deviation at 68%
std::pair<std::vector<double>, std::vector<double>> L1MetricPerOutput(Matrix const& real, Matrix const& predicted)
{
    std::vector<double> means(real[0].size(), 0);
    std::vector<double> dev(real[0].size(), 0);

    //mean absolute error
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            means[j] += std::abs(real[i][j] - predicted[i][j]);
        }
    }
    for(unsigned i = 0; i < real[0].size(); i++)
    {
        means[i] /= real.size();
    }

    //dev
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            dev[j] += std::pow(means[j] - std::abs(real[i][j] - predicted[i][j]), 2);
        }
    }
    for(unsigned i = 0; i < real[0].size(); i++)
    {
        dev[i] /= (real.size() - 1);
        dev[i] = std::sqrt(dev[i]);
    }

    return {means, dev};
}


//first is mse with normalized outputs, second is with unormalized ones
std::pair<double, double> L2Metric(Matrix const& real, Matrix const& predicted, std::vector<std::pair<double, double>> const& mM)
{
    std::vector<double> sums(real.size(), 0);
    std::vector<double> unormalizedSums(real.size(), 0);

    //mean square error
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            sums[i] += std::pow(real[i][j] - predicted[i][j], 2);
            unormalizedSums[i] += std::pow(real[i][j] - predicted[i][j], 2) * std::pow(mM[j].second - mM[j].first, 2);
        }
    }

    return {average(sums).first, average(unormalizedSums).first};
}


//first is mse (of normalized outputs), second is std deviation at 68%
std::pair<std::vector<double>, std::vector<double>> L2MetricPerOutput(Matrix const& real, Matrix const& predicted)
{
    std::vector<double> means(real[0].size(), 0);
    std::vector<double> dev(real[0].size(), 0);

    //mean square error
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            means[j] += std::pow(real[i][j] - predicted[i][j], 2);
        }
    }
    for(unsigned i = 0; i < real[0].size(); i++)
    {
        means[i] /= real.size();
    }

    //dev
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            dev[j] += std::pow(means[j] - std::pow(real[i][j] - predicted[i][j], 2), 2);
        }
    }
    for(unsigned i = 0; i < real[0].size(); i++)
    {
        dev[i] /= (real.size() - 1);
        dev[i] = std::sqrt(dev[i]);
    }

    return {means, dev};
}


} // namespace brain

#endif // BRAIN_TEST_HH_