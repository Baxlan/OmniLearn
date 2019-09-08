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


//first is "per feature", second is "per output"
std::pair<double, double> accuracy(Matrix const& real, Matrix const& predicted, double margin)
{
    double perFeatureUnvalidated = 0;
    double perOutputUnvalidated = 0;
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            //test "per output"
            if(real[i][j] > std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) < predicted[i][j] || predicted[i][j] < (real[i][j] * (1-margin/100)))
                {
                    perOutputUnvalidated++;
                }
            }
            else if(real[i][j] < -std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) > predicted[i][j] || predicted[i][j] > (real[i][j] * (1-margin/100)))
                {
                    perOutputUnvalidated++;
                }
            }
            else //if real == 0
            {
                if(false)
                {
                    perOutputUnvalidated++;
                }
            }
        }
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            //test "per feature"
            if(real[i][j] > std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) < predicted[i][j] || predicted[i][j] < (real[i][j] * (1-margin/100)))
                {
                    perFeatureUnvalidated++;
                    break;
                }
            }
            else if(real[i][j] < -std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) > predicted[i][j] || predicted[i][j] > (real[i][j] * (1-margin/100)))
                {
                    perFeatureUnvalidated++;
                    break;
                }
            }
            else //if real == 0
            {
                if(false)
                {
                    perFeatureUnvalidated++;
                    break;
                }
            }
        }
    }
    perFeatureUnvalidated = 100* (real.size() - perFeatureUnvalidated)/(real.size());
    perOutputUnvalidated = 100* (real.size()*real[0].size() - perOutputUnvalidated)/(real.size()*real[0].size());
    return {perFeatureUnvalidated, perOutputUnvalidated};
}


//first is "per feature", second is "per output"
std::vector<double> accuracyPerOutput(Matrix const& real, Matrix const& predicted, double margin)
{
    std::vector<double> unvalidated(real[0].size(), 0);
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            //test "per output"
            if(real[i][j] > std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) < predicted[i][j] || predicted[i][j] < (real[i][j] * (1-margin/100)))
                {
                    unvalidated[j]++;
                }
            }
            else if(real[i][j] < -std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) > predicted[i][j] || predicted[i][j] > (real[i][j] * (1-margin/100)))
                {
                    unvalidated[j]++;
                }
            }
            else //if real == 0
            {
                if(false)
                {
                    unvalidated[j]++;
                }
            }
        }
    }
    for(unsigned i = 0; i < unvalidated.size(); i++)
    {
        unvalidated[i] = 100* (real.size() - unvalidated[i])/(real.size());
    }
    return unvalidated;
}


} // namespace brain

#endif // BRAIN_TEST_HH_