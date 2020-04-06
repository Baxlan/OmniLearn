// decay.hh

#ifndef OMNILEARN_ANNEALING_HH_
#define OMNILEARN_ANNEALING_HH_

#include <cmath>



namespace omnilearn
{



double inverse(double initialLR, size_t epoch, double decayValue);
double exp(double initialLR, size_t epoch, double decayValue);
double step(double initialLR, size_t epoch, double decayValue, size_t delay);



} // namespace omnilearn



#endif // OMNILEARN_ANNEALING_HH_