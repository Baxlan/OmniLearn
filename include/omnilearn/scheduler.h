// decay.h

#ifndef OMNILEARN_SCHEDULER_H_
#define OMNILEARN_SCHEDULER_H_

#include <cmath>



namespace omnilearn
{



// learning rate schedule
double LRexp(double initialValue, size_t iteration, double decayValue);
double LRstep(double initialValue, size_t iteration, double decayValue, size_t delay);

// batch size schedule
double BSexp(double initialValue, size_t iteration, double decayValue);
double BSstep(double initialValue, size_t iteration, double decayValue, size_t delay);

// momentum scedule
double Mexp(double initialValue, double maxValue, size_t iteration, double growthValue);
double Mstep(double initialValue, double maxValue, size_t iteration, double growthValue, size_t delay);



} // namespace omnilearn



#endif // OMNILEARN_SCHEDULER_H_