// optimizer.cpp

#include "omnilearn/optimizer.h"



void omnilearn::optimizedUpdate(double& coefToUpdate, double& previousGrad, double& previousGrad2, double& optimalPreviousGrad2, double& previousUpdate, double gradient, bool automaticLearningRate, bool adaptiveLearningRate,
                                double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    gradient = gradient - (L2 * coefToUpdate) - (coefToUpdate > 0 ? L1 : -L1); // regularization
    previousGrad = (momentum * previousGrad) + ((1 - momentum) * gradient); // momentum

    // We update with the actual gradient ((1 - momentum) * gradient) whose the momentum have already been taken into account at the previous update,
    // and we apply the momentum of the next update now (nextMomentum * previousGrad) ==> nesterov acceletated gradient
    double gradientUpdate = ((1 - momentum) * gradient / (1 - cumulativeMomentum)) + (nextMomentum * previousGrad / (1 - (cumulativeMomentum * nextMomentum)));
    // the 1/(1-cumulativeMomentum) and 1/(1-(cumulativeMomentum * nextMomentum)) factors are here to unbias gradients at the first iterations

    previousGrad2 = (window * previousGrad2) + ((1 - window) * std::pow(gradient, 2)); // adaptive learning rate (window effect) on the denominator of the learning rate
    optimalPreviousGrad2 = std::max(previousGrad2, optimalPreviousGrad2 * (iteration > 1 ? std::pow((1-momentum) / (1-previousMomentum), 2) : 1));
    // the max operator is used to get the AMSGrad optimizer advantage on sparse data (i.e. if informative data are infrequent, this max operator gives them more power)
    // the std::pow((1-momentum) / (1-previousMomentum), 2) factor is here to get the AdamX optimizer behavior (AdamX is a slight correction of AMSGrad for mathematical purposes)

    learningRate = (automaticLearningRate ? std::sqrt(previousUpdate + optimizerBias) : learningRate); // adaptive learning rate (window effect) on the numerator of the learning rate
    learningRate /= (adaptiveLearningRate ? std::sqrt((optimalPreviousGrad2 / (1-std::pow(window, iteration))) + optimizerBias) : 1); // apply the denominator calculated previously
    // the 1/(1-std::pow(window, iteration)) factor is here to unbias optimalPreviousGrad2 at the first iterations

    double oldCoef = coefToUpdate;
    coefToUpdate += learningRate * (gradientUpdate - (decay * coefToUpdate));
    // decay is decoupled from gradient. See AdamW optimizer

    // the update of previousUpdate is performed after (because we don't have it before ...)
    previousUpdate = (window * previousUpdate) + ((1 - window) * std::pow(oldCoef - coefToUpdate, 2));
}