// decay.cpp

#include "omnilearn/scheduler.h"



double omnilearn::LRexp(double initialValue, size_t iteration, double decayValue)
{
    return initialValue * std::exp(-decayValue * static_cast<double>(iteration-1));
}


double omnilearn::LRstep(double initialValue, size_t iteration, double decayValue, size_t delay)
{
    return initialValue * std::pow(1/decayValue, std::floor((iteration-1)/delay));
}


double omnilearn::BSexp(double initialValue, size_t iteration, double growthValue)
{
    return LRexp(initialValue, iteration, -growthValue);
}


double omnilearn::BSstep(double initialValue, size_t iteration, double growthValue, size_t delay)
{
    return LRstep(initialValue, iteration, 1/growthValue, delay);
}


double omnilearn::Mexp(double initialValue, double maxValue, size_t iteration, double growthValue)
{
    return maxValue - LRexp(maxValue + initialValue, iteration, growthValue);
}


double omnilearn::Mstep(double initialValue, double maxValue, size_t iteration, double growthValue, size_t delay)
{
    return maxValue - LRstep(maxValue + initialValue, iteration, 1/growthValue, delay);
}