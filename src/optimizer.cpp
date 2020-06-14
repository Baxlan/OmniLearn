// optimizer.cpp

#include "omnilearn/optimizer.h"
#include <limits>



void omnilearn::optimizedUpdate(double& coefToUpdate, double& previousGrad, double& previousGrad2, double& optimalPreviousGrad2, double& previousUpdate, double gradient, bool automaticLearningRate,
                                bool adaptiveLearningRate, double learningRate, double momentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    // to disable momentum, set it to 0
    // to disable window effect, set it to 0

    // if momentum or window == 1, division by 0 occurs
    // if automaticLearningRate == false, window should be superior to 0

    gradient = gradient - (L2 * coefToUpdate) - (coefToUpdate > 0 ? L1 : -L1);

    previousGrad = (momentum * previousGrad) + ((1 - momentum) * gradient);

    previousGrad2 = (window * previousGrad2) + ((1 - window) * std::pow(gradient, 2));
    optimalPreviousGrad2 = (adaptiveLearningRate ? std::max(optimalPreviousGrad2, previousGrad2) : 1);
    // the max operator is used to get the AMSGrad optimizer advantage on sparse data (i.e. if informative data are infrequent, this max operator give them more power)

    learningRate = (automaticLearningRate ?  (std::sqrt(previousUpdate) + optimizerBias) : learningRate);
    learningRate /= (std::sqrt(optimalPreviousGrad2 / (1-std::pow(window, iteration))) + optimizerBias);
    // the 1/(1-std::pow(window, iteration)) factor is here to unbias optimalPreviousGrad2 at the first iterations

    double oldCoef = coefToUpdate;
    coefToUpdate += learningRate * (previousGrad/(1-std::pow(momentum, iteration)) - (decay * coefToUpdate));
    // decay is decoupled from gradient. See AdamW optimizer
    // the 1/(1-std::pow(momentum, iteration)) factor is here to unbias previousGrad at the first iterations

    // the update of previousUpdate is performed after (because we don't have it before ...)
    previousUpdate = (window * previousUpdate) + ((1 - window) * std::pow(oldCoef - coefToUpdate, 2));
}