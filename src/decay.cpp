// decay.cpp

#include "omnilearn/decay.hh"



double omnilearn::inverse(double initialLR, size_t epoch, double decayValue)
{
    return initialLR / (1 + (decayValue * static_cast<double>(epoch-1)));
}


double omnilearn::exp(double initialLR, size_t epoch, double decayValue)
{
    return initialLR * std::exp(-decayValue * static_cast<double>(epoch-1));
}


double omnilearn::step(double initialLR, size_t epoch, double decayValue, size_t delay)
{
    return initialLR * std::pow(decayValue, std::floor((epoch-1)/delay));
}