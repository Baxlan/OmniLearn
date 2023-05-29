// optimizer.h

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <cmath> // included here (not in .cpp) because size_t is needed



namespace omnilearn
{



struct LearnableParameterInfos
{
    LearnableParameterInfos():
    gradient(0),
    previousGrad(0),
    previousGrad2(0),
    optimalPreviousGrad2(0),
    previousUpdate(0)
    {
    }

    double gradient; //sum (over features of the batch) of partial gradient of aggregation according to each weight
    double previousGrad; // momentum effect
    double previousGrad2; // window effect
    double optimalPreviousGrad2; // see AMSGrad documentation
    double previousUpdate; // replaces learning rate if asked
};



void optimizedUpdate(double& coefToUpdate, LearnableParameterInfos& parameterInfos, bool automaticLearningRate, bool adaptiveLearningRate,
                    bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum,
                    double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2,
                    double decay, bool avoidZero = false);



} // namespace omnilearn



#endif // OPTIMIZER_H_