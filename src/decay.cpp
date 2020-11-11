// decay.cpp

#include "omnilearn/decay.h"



double omnilearn::inverse(double initialValue, size_t epoch, double decayValue)
{
    return initialValue / (1 + (decayValue * static_cast<double>(epoch-1)));
}


double omnilearn::exp(double initialValue, size_t epoch, double decayValue)
{
    return initialValue * std::exp(-decayValue * static_cast<double>(epoch-1));
}


double omnilearn::step(double initialValue, size_t epoch, double decayValue, size_t delay)
{
    return initialValue * std::pow(decayValue, std::floor((epoch-1)/delay));
}


double omnilearn::growingInverse(double initialValue, double maxValue, size_t epoch, double growthValue)
{
    return maxValue - inverse(maxValue + initialValue, epoch, growthValue);
}


double omnilearn::growingExp(double initialValue, double maxValue, size_t epoch, double growthValue)
{
    return maxValue - exp(maxValue + initialValue, epoch, growthValue);
}


double omnilearn::growingStep(double initialValue, double maxValue, size_t epoch, double growthValue, size_t delay)
{
    return maxValue - step(maxValue + initialValue, epoch, growthValue, delay);
}