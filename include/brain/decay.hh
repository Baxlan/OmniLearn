#ifndef BRAIN_ANNEALING_HH_
#define BRAIN_ANNEALING_HH_

#include <cmath>

namespace brain
{



double inverse(double initialLR, size_t epoch, double decayValue)
{
    return initialLR / (1 + (decayValue * static_cast<double>(epoch-1)));
}


double exp(double initialLR, size_t epoch, double decayValue)
{
    return initialLR * std::exp(-decayValue * static_cast<double>(epoch-1));
}


double step(double initialLR, size_t epoch, double decayValue, size_t delay)
{
    return initialLR * std::pow(decayValue, std::floor((epoch-1)/delay));
}



} // namespace brain

#endif // BRAIN_ANNEALING_HH_