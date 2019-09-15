#ifndef BRAIN_TEST_HH_
#define BRAIN_TEST_HH_

#include "vectorial.hh"

namespace brain
{


// the inputs are loss, the output is average loss
double averageLoss(Matrix const& loss)
{
    std::vector<double> feature(loss.size());
    for(unsigned i = 0; i < loss.size(); i++)
    {
        feature[i] = sum(loss[i]);
    }
    return average(feature).first;
}


//first is "accuracy", second is "false positive"
std::pair<double, double> accuracy(Matrix const& real, Matrix const& predicted, double classValidity)
{
    double validated = 0;
    double fp = 0; //false positive

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


//first is "accuracy", second is "false positive"
std::pair<std::vector<double>, std::vector<double>> accuracyPerOutput(Matrix const& real, Matrix const& predicted, double classValidity)
{
    std::vector<double> validated(real[0].size(), 0);
    std::vector<double> fp(real[0].size(), 0); //false positive
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


} // namespace brain

#endif // BRAIN_TEST_HH_