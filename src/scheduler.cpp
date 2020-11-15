// decay.cpp

#include "omnilearn/scheduler.h"



double omnilearn::LRexp(double initialValue, size_t epoch, double decayValue)
{
    return initialValue * std::exp(-decayValue * static_cast<double>(epoch-1));
}


double omnilearn::LRstep(double initialValue, size_t epoch, double decayValue, size_t delay)
{
    return initialValue * std::pow(1/decayValue, std::floor((epoch-1)/delay));
}


double omnilearn::BSexp(double initialValue, size_t epoch, double growthValue)
{
    return LRexp(initialValue, epoch, -growthValue);
}


double omnilearn::BSstep(double initialValue, size_t epoch, double growthValue, size_t delay)
{
    return LRstep(initialValue, epoch, 1/growthValue, delay);
}


double omnilearn::Mexp(double initialValue, double maxValue, size_t epoch, double growthValue)
{
    return maxValue - LRexp(maxValue + initialValue, epoch, growthValue);
}


double omnilearn::Mstep(double initialValue, double maxValue, size_t epoch, double growthValue, size_t delay)
{
    return maxValue - LRstep(maxValue + initialValue, epoch, 1/growthValue, delay);
}