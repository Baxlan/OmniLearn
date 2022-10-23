// optimizer.cpp

#include "omnilearn/optimizer.h"



void omnilearn::optimizedUpdate(double& coefToUpdate, double& previousGrad, double& previousGrad2, double& optimalPreviousGrad2, double& previousUpdate, double gradient, bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator,
                                double learningRate, double momentum, [[maybe_unused]] double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay, bool avoidZero)
{
    gradient = gradient - (L2 * coefToUpdate) - (coefToUpdate > 0 ? L1 : -L1); // regularization
    previousGrad = (momentum * previousGrad) + ((1 - momentum) * gradient); // momentum (FIRST MOMENT OF GRADIENT)

    // We update with the current gradient ((1 - momentum) * gradient) whose the momentum have already been taken into account at the previous update,
    // and we apply the momentum of the next update now (nextMomentum * previousGrad) ==> it's the nesterov acceletated gradient
    double gradientUpdate = ((1 - momentum) * gradient / (1 - cumulativeMomentum)) + (nextMomentum * previousGrad / (1 - (cumulativeMomentum * nextMomentum)));
    // the 1/(1-cumulativeMomentum) and 1/(1-(cumulativeMomentum * nextMomentum)) factors are here to unbias gradients at the first iterations

    // adaptive learning rate by exponential decay of past squared GRADIENTS (SECOND MOMENT OF GRADIENT)
    previousGrad2 = (window * previousGrad2) + ((1 - window) * std::pow(gradient, 2) / (1-std::pow(window, iteration)));
    optimalPreviousGrad2 = (useMaxDenominator ? std::max(previousGrad2, optimalPreviousGrad2) : previousGrad2);
    // the max operator is used to get AMSGrad : it forbids the LR to grow up (unlike RMSProp/Adam, leading to non convergence),
    // and the LR doesn't become infinitesimal thank to the winow effect of exponential decay (unlike Adagrad)
    // the 1/(1-std::pow(window, iteration)) factor is here to unbias optimalPreviousGrad2 at the first iterations

    // adaptive learning rate by exponential decay of past squared UPDATES (SECOND MOMENT OF UPDATES) ==> Adadelta
    // (this is based on a second order method : the Newton's method with Hessian approximation)
    learningRate = (automaticLearningRate ? std::sqrt((previousUpdate) + optimizerBias) : learningRate);
    // we must use the previous update, not the current one, because we don't know the curent one yet...
    // this is an approximation where we assume that the curvature of the loss function is locally smooth

    // then applying the exponential decay previously calculated
    learningRate /= (adaptiveLearningRate ? std::sqrt((optimalPreviousGrad2) + optimizerBias) : 1);

    double oldCoef = coefToUpdate;
    coefToUpdate += learningRate * (gradientUpdate - (decay * coefToUpdate));
    // decay is decoupled from gradient. See AdamW optimizer

    // avoid coefToUpdate being 0 if it is used as denominator somewhere
    if(avoidZero && std::abs(coefToUpdate) < 1e-4)
        coefToUpdate = (coefToUpdate < 0 ? -1e-4: 1e-4);

    // now we know the current update, so we can update "previousUpdate"
    previousUpdate = (window * previousUpdate) + ((1 - window) * std::pow(oldCoef - coefToUpdate, 2) / (1-std::pow(window, iteration)));
    // the 1/(1-std::pow(window, iteration)) is here to unbias the second moment of updates. That haven't been found in any paper so this is an innovation here.
}