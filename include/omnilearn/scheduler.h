// decay.h

#ifndef OMNILEARN_SCHEDULER_H_
#define OMNILEARN_SCHEDULER_H_

#include <cmath>



namespace omnilearn
{



double inverse(double initialValue, size_t epoch, double decayValue);
double exp(double initialValue, size_t epoch, double decayValue);
double step(double initialValue, size_t epoch, double decayValue, size_t delay);

double growingInverse(double initialValue, double maxValue, size_t epoch, double growthValue);
double growingExp(double initialValue, double maxValue, size_t epoch, double growthValue);
double growingStep(double initialValue, double maxValue, size_t epoch, double growthValue, size_t delay);



} // namespace omnilearn



#endif // OMNILEARN_SCHEDULER_H_