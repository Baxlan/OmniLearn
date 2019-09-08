#ifndef BRAIN_ANNEALING_HH_
#define BRAIN_ANNEALING_HH_

#include <cmath>

namespace brain
{


namespace LRDecay
{


double none(double initialLR, [[maybe_unused]] unsigned epoch, [[maybe_unused]] double decayConstant, [[maybe_unused]] unsigned step)
{
    return initialLR;
}


double inverse(double initialLR, unsigned epoch, double decayConstant, [[maybe_unused]] unsigned step)
{
    return initialLR / (1 + (decayConstant * (epoch-1)));
}


double exp(double initialLR, unsigned epoch, double decayConstant, [[maybe_unused]] unsigned step)
{
    return initialLR * std::exp(-decayConstant * (epoch-1));
}


double step(double initialLR, unsigned epoch, double decayConstant, unsigned step)
{
    return initialLR * std::pow(decayConstant, std::floor((epoch-1)/step));
}


} // namespace LRDecay


} // namespace brain

#endif // BRAIN_ANNEALING_HH_