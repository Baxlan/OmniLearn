#ifndef BRAIN_ANNEALING_HH_
#define BRAIN_ANNEALING_HH_

#include <cmath>

namespace brain
{


namespace decay
{


double none(double initialLR, [[maybe_unused]] unsigned epoch, [[maybe_unused]] double decayValue, [[maybe_unused]] unsigned delay)
{
    return initialLR;
}


double inverse(double initialLR, unsigned epoch, double decayValue, [[maybe_unused]] unsigned delay)
{
    return initialLR / (1 + (decayValue * (epoch-1)));
}


double exp(double initialLR, unsigned epoch, double decayValue, [[maybe_unused]] unsigned delay)
{
    return initialLR * std::exp(-decayValue * (epoch-1));
}


double step(double initialLR, unsigned epoch, double decayValue, unsigned delay)
{
    return initialLR * std::pow(decayValue, std::floor((epoch-1)/delay));
}


} // namespace LRDecay


} // namespace brain

#endif // BRAIN_ANNEALING_HH_