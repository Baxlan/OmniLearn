// optimizer.h

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <cmath> // included here (not in .cpp) because size_t is needed



namespace omnilearn
{



void optimizedUpdate(double& coefToUpdate, double& previousGrad, double& previousGrad2, double& optimalPreviousGrad2,
                    double& previousUpdate, double gradient, bool automaticLearningRate, bool adaptiveLearningRate,
                    bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum,
                    double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2,
                    double decay, bool avoidZero = false);



} // namespace omnilearn



#endif // OPTIMIZER_H_