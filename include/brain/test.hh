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

    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            if(std::abs(real[i][j] - 1) <= std::numeric_limits<double>::epsilon())
            {
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
    validated = 100*validated/(real.size());
    fp = 100*fp/(real.size());
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
        fp[i] = 100*fp[i]/(count[i]+fp[i]);
    }
    return {validated, fp};
}



//first is "mean", second is "std deviation at 68%"
std::pair<double, double> L1Cost(Matrix const& real, Matrix const& predicted)
{
    std::vector<double> sums(real.size(), 0);

    //mean absolute error
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            sums[i] += std::abs(real[i][j] - predicted[i][j]);
        }
    }

    return average(sums);
}


//first is "mean", second is "std deviation at 68%"
std::pair<std::vector<double>, std::vector<double>> L1CostPerOutput(Matrix const& real, Matrix const& predicted)
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
        dev[i] /= real.size();
        dev[i] = std::sqrt(dev[i]);
    }

    return {means, dev};
}


//first is "mean", second is "std deviation at 68%"
std::pair<double, double> L2Cost(Matrix const& real, Matrix const& predicted)
{
    std::vector<double> sums(real.size(), 0);

    //mean absolute error
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            sums[i] += 0.5 * std::pow(real[i][j] - predicted[i][j], 2);
        }
    }

    return average(sums);
}


//first is "mean", second is "std deviation at 68%"
std::pair<std::vector<double>, std::vector<double>> L2CostPerOutput(Matrix const& real, Matrix const& predicted)
{
    std::vector<double> means(real[0].size(), 0);
    std::vector<double> dev(real[0].size(), 0);

    //mean squared error
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
        dev[i] /= real.size();
        dev[i] = std::sqrt(dev[i]);
    }

    return {means, dev};
}


} // namespace brain

#endif // BRAIN_TEST_HH_