// Activation.cpp

#include "omnilearn/Activation.hh"
#include "omnilearn/Exception.hh"
#include "omnilearn/optimizer.h"



//=============================================================================
//=============================================================================
//=============================================================================
//=== LINEAR ACTIVATION =======================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Linear::Linear(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Linear activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


double omnilearn::Linear::activate(double val) const
{
    return val;
}


double omnilearn::Linear::prime([[maybe_unused]] double val) const
{
    return 1;
}


void omnilearn::Linear::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Linear::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Linear::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Linear activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Linear::getCoefs() const
{
    return Vector(0);
}

//static function
omnilearn::Activation omnilearn::Linear::signature() const
{
    return Activation::Linear;
}


void omnilearn::Linear::keep()
{
    //nothing to do
}


void omnilearn::Linear::release()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== SIGMOID ACTIVATION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================




omnilearn::Sigmoid::Sigmoid(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Sigmoid activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


double omnilearn::Sigmoid::activate(double val) const
{
    return 1 / (1 + std::exp(-val));
}


double omnilearn::Sigmoid::prime(double val) const
{
    double val2 = activate(val);
    return val2 * (1 - val2);
}


void omnilearn::Sigmoid::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Sigmoid::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Sigmoid::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Sigmoid activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Sigmoid::getCoefs() const
{
    return Vector(0);
}


omnilearn::Activation omnilearn::Sigmoid::signature() const
{
    return Activation::Sigmoid;
}


void omnilearn::Sigmoid::keep()
{
    //nothing to do
}


void omnilearn::Sigmoid::release()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== TANH ACTIVATION =========================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Tanh::Tanh(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Tanh activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


double omnilearn::Tanh::activate(double val) const
{
    return std::tanh(val);
}


double omnilearn::Tanh::prime(double val) const
{
    return 1 - std::pow(activate(val),2);
}


void omnilearn::Tanh::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Tanh::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Tanh::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Tanh activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Tanh::getCoefs() const
{
    return Vector(0);
}


omnilearn::Activation omnilearn::Tanh::signature() const
{
    return Activation::Tanh;
}


void omnilearn::Tanh::keep()
{
    //nothing to do
}


void omnilearn::Tanh::release()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== SOFTPLUS ACTIVATION =====================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Softplus::Softplus(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Softplus activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


double omnilearn::Softplus::activate(double val) const
{
    return std::log(std::exp(val) + 1);
}


double omnilearn::Softplus::prime(double val) const
{
    return 1 / (1 + std::exp(-val));
}


void omnilearn::Softplus::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Softplus::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Softplus::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Softplus activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Softplus::getCoefs() const
{
    return Vector(0);
}


omnilearn::Activation omnilearn::Softplus::signature() const
{
    return Activation::Softplus;
}


void omnilearn::Softplus::keep()
{
    //nothing to do
}


void omnilearn::Softplus::release()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== (LEAKY) RELU ACTIVATION =================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Relu::Relu(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Relu/Prelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
}


double omnilearn::Relu::activate(double val) const
{
    return (val < 0 ? _coef*val : val);
}


double omnilearn::Relu::prime(double val) const
{
    return (val < 0 ? _coef : 1);
}


void omnilearn::Relu::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Relu::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Relu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Relu/Prelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
}


omnilearn::rowVector omnilearn::Relu::getCoefs() const
{
    return (Vector(1) << _coef).finished();
}


omnilearn::Activation omnilearn::Relu::signature() const
{
    return Activation::Relu;
}


void omnilearn::Relu::keep()
{
    _savedCoef = _coef;
}


void omnilearn::Relu::release()
{
    _coef = _savedCoef;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC (LEAKY) RELU ACTIVATION ======================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Prelu::Prelu(Vector const& coefs):
Relu(coefs),
_coefGradient(0),
_previousCoefGrad(0),
_previousCoefGrad2(0),
_optimalPreviousCoefGrad2(0),
_previousCoefUpdate(0),
_counter(0)
{
}


void omnilearn::Prelu::computeGradients(double aggr, double inputGrad)
{
    _coefGradient -= inputGrad * (aggr < 0 ? aggr : 0);
    _counter += (aggr < 0 ? 1 : 0);
}


void omnilearn::Prelu::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    if(_counter == 0)
        return;

    _coefGradient /= static_cast<double>(_counter);

    optimizedUpdate(_coef, _previousCoefGrad, _previousCoefGrad2, _optimalPreviousCoefGrad2, _previousCoefUpdate, _coefGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    _coefGradient = 0;
    _counter = 0;
}


void omnilearn::Prelu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Relu/Prelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
    _coefGradient = 0;
    _previousCoefGrad = 0;
    _previousCoefGrad2 = 0;
    _optimalPreviousCoefGrad2 = 0;
    _previousCoefUpdate = 0;
    _counter = 0;
}


omnilearn::Activation omnilearn::Prelu::signature() const
{
    return Activation::Prelu;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== EXPONENTIAL RELU ACTIVATION =============================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Elu::Elu(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Elu/Pelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
}


double omnilearn::Elu::activate(double val) const
{
    return (val < 0 ? _coef*(std::exp(val)-1) : val);
}


double omnilearn::Elu::prime(double val) const
{
    return (val < 0 ? _coef * std::exp(val) : 1);
}


void omnilearn::Elu::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Elu::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Elu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Elu/Pelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
}


omnilearn::rowVector omnilearn::Elu::getCoefs() const
{
    return (Vector(1) << _coef).finished();
}


omnilearn::Activation omnilearn::Elu::signature() const
{
    return Activation::Elu;
}


void omnilearn::Elu::keep()
{
    _savedCoef = _coef;
}


void omnilearn::Elu::release()
{
    _coef = _savedCoef;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC EXPONENTIAL RELU ACTIVATION ==================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Pelu::Pelu(Vector const& coefs):
Elu(coefs),
_coefGradient(0),
_previousCoefGrad(0),
_previousCoefGrad2(0),
_optimalPreviousCoefGrad2(0),
_previousCoefUpdate(0),
_counter(0)
{
}


void omnilearn::Pelu::computeGradients(double aggr, double inputGrad)
{
    _coefGradient -= inputGrad * (aggr < 0 ? std::exp(aggr)-1 : 0);
    _counter += (aggr < 0 ? 1 : 0);
}


void omnilearn::Pelu::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    if(_counter == 0)
        return;

    _coefGradient /= static_cast<double>(_counter);

    optimizedUpdate(_coef, _previousCoefGrad, _previousCoefGrad2, _optimalPreviousCoefGrad2, _previousCoefUpdate, _coefGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    _coefGradient = 0;
    _counter = 0;
}


void omnilearn::Pelu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Elu/Pelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
    _coefGradient = 0;
    _previousCoefGrad = 0;
    _previousCoefGrad2 = 0;
    _optimalPreviousCoefGrad2 = 0;
    _previousCoefUpdate = 0;
    _counter = 0;
}


omnilearn::Activation omnilearn::Pelu::signature() const
{
    return Activation::Pelu;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== (PARAMETRIC) S-SHAPED ACTIVATION ========================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Srelu::Srelu(Vector const& coefs)
{
    if(coefs.size() != 4)
        throw Exception("Srelu activation function needs 4 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _coef1 = coefs[0];
    _coef2 = coefs[1];
    _hinge1 = coefs[2];
    _hinge2 = coefs[3];

    _savedCoef1 = 0;
    _savedCoef2 = 0;
    _savedHinge1 = 0;
    _savedHinge2 = 0;

    _coef1Gradient = 0;
    _coef2Gradient = 0;
    _hinge1Gradient = 0;
    _hinge2Gradient = 0;

    _previousCoef1Grad = 0;
    _previousCoef2Grad = 0;
    _previousHinge1Grad = 0;
    _previousHinge2Grad = 0;

    _previousCoef1Grad2 = 0;
    _previousCoef2Grad2 = 0;
    _previousHinge1Grad2 = 0;
    _previousHinge2Grad2 = 0;

    _optimalPreviousCoef1Grad2 = 0;
    _optimalPreviousCoef2Grad2 = 0;
    _optimalPreviousHinge1Grad2 = 0;
    _optimalPreviousHinge2Grad2 = 0;

    _previousCoef1Update = 0;
    _previousCoef2Update = 0;
    _previousHinge1Update = 0;
    _previousHinge2Update = 0;

    _counter1 = 0;
    _counter2 = 0;
}


double omnilearn::Srelu::activate(double val) const
{
    if(val <= _hinge1)
        return _hinge1 + _coef1 * (val - _hinge1);
    else if(val >= _hinge2)
        return _hinge2 + _coef2 * (val - _hinge2);
    else
        return val;
}


double omnilearn::Srelu::prime(double val) const
{
    if(val <= _hinge1)
        return _coef1;
    else if(val >= _hinge2)
        return _coef2;
    else
        return 1;
}


void omnilearn::Srelu::computeGradients(double aggr, double inputGrad)
{
    if(aggr <= _hinge1)
    {
        _coef1Gradient -= inputGrad * (aggr - _hinge1);
        _hinge1Gradient -= inputGrad * (1 - _coef1);
        _counter1 += 1;
    }
    else if(aggr >= _hinge2)
    {
        _coef2Gradient += inputGrad * (aggr - _hinge2);
        _hinge2Gradient += inputGrad * (1 - _coef2);
        _counter2 += 1;
    }
}


void omnilearn::Srelu::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _coef1Gradient /= static_cast<double>(_counter1);
    _coef2Gradient /= static_cast<double>(_counter2);
    _hinge1Gradient /= static_cast<double>(_counter1);
    _hinge2Gradient /= static_cast<double>(_counter2);

    optimizedUpdate(_coef1, _previousCoef1Grad, _previousCoef1Grad2, _optimalPreviousCoef1Grad2, _previousCoef1Update, _coef1Gradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_coef2, _previousCoef2Grad, _previousCoef2Grad2, _optimalPreviousCoef2Grad2, _previousCoef2Update, _coef2Gradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_hinge1, _previousHinge1Grad, _previousHinge1Grad2, _optimalPreviousHinge1Grad2, _previousHinge1Update, _hinge1Gradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_hinge2, _previousHinge2Grad, _previousHinge2Grad2, _optimalPreviousHinge2Grad2, _previousHinge2Update, _hinge2Gradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    _coef1Gradient = 0;
    _coef2Gradient = 0;
    _hinge1Gradient = 0;
    _hinge2Gradient = 0;
    _counter1 = 0;
    _counter2 = 0;
}


void omnilearn::Srelu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 4)
        throw Exception("Srelu activation function needs 4 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _coef1 = coefs[0];
    _coef2 = coefs[1];
    _hinge1 = coefs[3];
    _hinge2 = coefs[4];

    _savedCoef1 = 0;
    _savedCoef2 = 0;
    _savedHinge1 = 0;
    _savedHinge2 = 0;

    _coef1Gradient = 0;
    _coef2Gradient = 0;
    _hinge1Gradient = 0;
    _hinge2Gradient = 0;

    _previousCoef1Grad = 0;
    _previousCoef2Grad = 0;
    _previousHinge1Grad = 0;
    _previousHinge2Grad = 0;

    _previousCoef1Grad2 = 0;
    _previousCoef2Grad2 = 0;
    _previousHinge1Grad2 = 0;
    _previousHinge2Grad2 = 0;

    _optimalPreviousCoef1Grad2 = 0;
    _optimalPreviousCoef2Grad2 = 0;
    _optimalPreviousHinge1Grad2 = 0;
    _optimalPreviousHinge2Grad2 = 0;

    _previousCoef1Update = 0;
    _previousCoef2Update = 0;
    _previousHinge1Update = 0;
    _previousHinge2Update = 0;

    _counter1 = 0;
    _counter2 = 0;
}


omnilearn::rowVector omnilearn::Srelu::getCoefs() const
{
    return (Vector(4) << _coef1, _coef2, _hinge1, _hinge2).finished();
}


omnilearn::Activation omnilearn::Srelu::signature() const
{
    return Activation::Srelu;
}


void omnilearn::Srelu::keep()
{
    _savedCoef1 = _coef1;
    _savedCoef2 = _coef2;
    _savedHinge1 = _hinge1;
    _savedHinge2 = _hinge2;
}


void omnilearn::Srelu::release()
{
    _coef1 = _savedCoef1;
    _coef2 = _savedCoef2;
    _hinge1 = _savedHinge1;
    _hinge2 = _savedHinge2;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== GAUSS ACTIVATION ========================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Gauss::Gauss(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Gauss/Pgauss activation functions need 2 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _center = coefs[0];
    _dev = coefs[1];

    _savedCenter = 0;
    _savedDev = 0;
}


double omnilearn::Gauss::activate(double val) const
{
    return std::exp(-std::pow(val - _center, 2) / (2*std::pow(_dev, 2)));
}


double omnilearn::Gauss::prime(double val) const
{
    return activate(val) * (_center - val) / std::pow(_dev, 2);
}


void omnilearn::Gauss::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Gauss::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Gauss::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Gauss/Pgauss activation functions need 2 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _center = coefs[0];
    _dev = coefs[1];

    _savedCenter = 0;
    _savedDev = 0;
}


omnilearn::rowVector omnilearn::Gauss::getCoefs() const
{
    return (Vector(2) << _center, _dev).finished();
}


omnilearn::Activation omnilearn::Gauss::signature() const
{
    return Activation::Gauss;
}


void omnilearn::Gauss::keep()
{
    _savedCenter = _center;
    _savedDev = _dev;
}


void omnilearn::Gauss::release()
{
    _center = _savedCenter;
    _dev = _savedDev;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC GAUSS ACTIVATION =============================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Pgauss::Pgauss(Vector const& coefs):
Gauss(coefs),
_centerGradient(0),
_previousCenterGrad(0),
_previousCenterGrad2(0),
_optimalPreviousCenterGrad2(0),
_previousCenterUpdate(0),
_devGradient(0),
_previousDevGrad(0),
_previousDevGrad2(0),
_optimalPreviousDevGrad2(0),
_previousDevUpdate(0),
_counter(0)
{
}


void omnilearn::Pgauss::computeGradients(double aggr, double inputGrad)
{
    _centerGradient += inputGrad * (activate(aggr) * (aggr - _center) / std::pow(_dev, 2));
    _devGradient += inputGrad * (activate(aggr) * std::pow((aggr - _center), 2) / std::pow(_dev, 3));
    _counter += 1;
}


void omnilearn::Pgauss::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _centerGradient /= static_cast<double>(_counter);
    _devGradient /= static_cast<double>(_counter);

    optimizedUpdate(_center, _previousCenterGrad, _previousCenterGrad2, _optimalPreviousCenterGrad2, _previousCenterUpdate, _centerGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_dev, _previousDevGrad, _previousDevGrad2, _optimalPreviousDevGrad2, _previousDevUpdate, _devGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    //avoid division by 0 in "computeGradients()"
    if(std::abs(_dev) < 1e-3)
        _dev = (_dev < 0 ? -1e-3 : 1e-3);

    _centerGradient = 0;
    _devGradient = 0;
    _counter = 0;
}


void omnilearn::Pgauss::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Gauss/Pgauss activation functions need 2 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _center = coefs[0];
    _dev = coefs[1];

    _savedCenter = 0;
    _savedDev = 0;

    _centerGradient = 0;
    _previousCenterGrad = 0;
    _previousCenterGrad2 = 0;
    _optimalPreviousCenterGrad2 = 0;
    _previousCenterUpdate = 0;

    _devGradient = 0;
    _previousDevGrad = 0;
    _previousDevGrad2 = 0;
    _optimalPreviousDevGrad2 = 0;
    _previousDevUpdate = 0;

    _counter = 0;
}


omnilearn::Activation omnilearn::Pgauss::signature() const
{
    return Activation::Pgauss;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== SOFTEXP ACTIVATION (PARAMETRIC) =========================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Softexp::Softexp(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Softexp activation function needs 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;

    _coefGradient = 0;
    _previousCoefGrad = 0;
    _previousCoefGrad2 = 0;
    _optimalPreviousCoefGrad2 = 0;
    _previousCoefUpdate = 0;
    _counter = 0;
}


double omnilearn::Softexp::activate(double val) const
{
    if(_coef < -std::numeric_limits<double>::epsilon())
        return -std::log(1-(_coef*(val + _coef))) / _coef;
    else if(_coef > std::numeric_limits<double>::epsilon())
        return ((std::exp(_coef * val) - 1) / _coef) + _coef;
    else
        return val;
}


double omnilearn::Softexp::prime(double val) const
{
    if(_coef < 0)
        return 1 / (1 - (_coef * (_coef + val)));
    else
       return std::exp(_coef * val);
}


void omnilearn::Softexp::computeGradients(double aggr, double inputGrad)
{
    if(_coef < -std::numeric_limits<double>::epsilon())
        _coefGradient += inputGrad * (std::log(1-(_coef*(_coef+aggr))) - (_coef*(2*_coef + aggr)) / (_coef*(_coef+aggr)-1)) / std::pow(_coef, 2);
    else if(_coef > std::numeric_limits<double>::epsilon())
        _coefGradient += inputGrad * (std::pow(_coef, 2) + (_coef*aggr - 1)*std::exp(_coef*aggr) + 1) / std::pow(_coef, 2);
    else
        _coefGradient += inputGrad * (std::pow(aggr, 2) / 2) + 1;

    _counter += 1;
}


void omnilearn::Softexp::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _coefGradient /= static_cast<double>(_counter);

    optimizedUpdate(_coef, _previousCoefGrad, _previousCoefGrad2, _optimalPreviousCoefGrad2, _previousCoefUpdate, _coefGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    _coefGradient = 0;
    _counter = 0;
}


void omnilearn::Softexp::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Softexp activation function needs 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;

    _coefGradient = 0;
    _previousCoefGrad = 0;
    _previousCoefGrad2 = 0;
    _optimalPreviousCoefGrad2 = 0;
    _previousCoefUpdate = 0;
    _counter = 0;
}


omnilearn::rowVector omnilearn::Softexp::getCoefs() const
{
    return (Vector(1) << _coef).finished();
}


omnilearn::Activation omnilearn::Softexp::signature() const
{
    return Activation::Softexp;
}


void omnilearn::Softexp::keep()
{
    _savedCoef = _coef;
}


void omnilearn::Softexp::release()
{
    _coef = _savedCoef;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== SINUS ACTIVATION ========================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Sin::Sin(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Sin/Psin activation functions need 2 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _pulsation = coefs[0];
    _phase = coefs[1];
    _savedPulsation = 0;
    _savedPhase = 0;
}


double omnilearn::Sin::activate(double val) const
{
    return std::sin(_pulsation*val + _phase);
}


double omnilearn::Sin::prime(double val) const
{
    return _pulsation * std::cos(_pulsation*val + _phase);
}


void omnilearn::Sin::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Sin::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Sin::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Sin/Psin activation functions need 2 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _pulsation = coefs[0];
    _phase = coefs[1];
    _savedPulsation = 0;
    _savedPhase = 0;
}


omnilearn::rowVector omnilearn::Sin::getCoefs() const
{
    return (Vector(2) << _pulsation, _phase).finished();
}

//static function
omnilearn::Activation omnilearn::Sin::signature() const
{
    return Activation::Sin;
}


void omnilearn::Sin::keep()
{
    _savedPulsation = _pulsation;
    _savedPhase = _phase;
}


void omnilearn::Sin::release()
{
    _pulsation = _savedPulsation;
    _phase = _savedPhase;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC SINUS ACTIVATION =============================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Psin::Psin(Vector const& coefs):
Sin(coefs),
_pulsationGradient(0),
_previousPulsationGrad(0),
_previousPulsationGrad2(0),
_optimalPreviousPulsationGrad2(0),
_previousPulsationUpdate(0),
_phaseGradient(0),
_previousPhaseGrad(0),
_previousPhaseGrad2(0),
_optimalPreviousPhaseGrad2(0),
_previousPhaseUpdate(0),
_counter(0)
{
}


void omnilearn::Psin::computeGradients(double aggr, double inputGrad)
{
    _pulsationGradient += inputGrad * aggr * std::cos(_pulsation*aggr + _phase);
    _phaseGradient += inputGrad * std::cos(_pulsation*aggr + _phase);
    _counter++;
}


void omnilearn::Psin::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _pulsationGradient /= static_cast<double>(_counter);
    _phaseGradient /= static_cast<double>(_counter);

    optimizedUpdate(_pulsation, _previousPulsationGrad, _previousPulsationGrad2, _optimalPreviousPulsationGrad2, _previousPulsationUpdate, _pulsationGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_phase, _previousPhaseGrad, _previousPhaseGrad2, _optimalPreviousPhaseGrad2, _previousPhaseUpdate, _phaseGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    _pulsationGradient = 0;
    _phaseGradient = 0;
    _counter = 0;
}


void omnilearn::Psin::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Sin/Psin activation functions need 2 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _pulsation = coefs[0];
    _phase = coefs[1];
    _pulsationGradient = 0;
    _previousPulsationGrad = 0;
    _previousPulsationGrad2 = 0;
    _optimalPreviousPulsationGrad2 = 0;
    _previousPulsationUpdate = 0;
    _phaseGradient = 0;
    _previousPhaseGrad = 0;
    _previousPhaseGrad2 = 0;
    _optimalPreviousPhaseGrad2 = 0;
    _previousPhaseUpdate = 0;
    _counter = 0;
}


omnilearn::Activation omnilearn::Psin::signature() const
{
    return Activation::Psin;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== CARDINAL SINUS ACTIVATION ===============================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Sinc::Sinc(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Sinc/Psinc activation functions need 2 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _pulsation = coefs[0];
    _phase = coefs[1];
    _savedPulsation = 0;
    _savedPhase = 0;
}


double omnilearn::Sinc::activate(double val) const
{
    if(val > -std::numeric_limits<double>::epsilon() && val < std::numeric_limits<double>::epsilon())
        return 1;
    else
        return std::sin(_pulsation*val + _phase)/(_pulsation*val + _phase);
}


double omnilearn::Sinc::prime(double val) const
{
    if(val > -std::numeric_limits<double>::epsilon() && val < std::numeric_limits<double>::epsilon())
        return 0;
    else
        return _pulsation * (std::cos(_pulsation*val + _phase)*(_pulsation*val + _phase) - std::sin(_pulsation*val + _phase)) / std::pow(_pulsation*val + _phase, 2);
}


void omnilearn::Sinc::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Sinc::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Sinc::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Sinc/Psinc activation functions need 2 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _pulsation = coefs[0];
    _phase = coefs[1];
    _savedPulsation = 0;
    _savedPhase = 0;
}


omnilearn::rowVector omnilearn::Sinc::getCoefs() const
{
    return (Vector(2) << _pulsation, _phase).finished();
}

//static function
omnilearn::Activation omnilearn::Sinc::signature() const
{
    return Activation::Sinc;
}


void omnilearn::Sinc::keep()
{
    _savedPulsation = _pulsation;
    _savedPhase = _phase;
}


void omnilearn::Sinc::release()
{
    _pulsation = _savedPulsation;
    _phase = _savedPhase;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC CARDINAL SINUS ACTIVATION ====================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Psinc::Psinc(Vector const& coefs):
Sinc(coefs),
_pulsationGradient(0),
_previousPulsationGrad(0),
_previousPulsationGrad2(0),
_optimalPreviousPulsationGrad2(0),
_previousPulsationUpdate(0),
_phaseGradient(0),
_previousPhaseGrad(0),
_previousPhaseGrad2(0),
_optimalPreviousPhaseGrad2(0),
_previousPhaseUpdate(0),
_counter(0)
{
}


void omnilearn::Psinc::computeGradients(double aggr, double inputGrad)
{
    _pulsationGradient += inputGrad * aggr * (std::cos(_pulsation*aggr + _phase)*(_pulsation*aggr + _phase) - std::sin(_pulsation*aggr + _phase)) / std::pow(_pulsation*aggr + _phase, 2);
    _phaseGradient += inputGrad * (std::cos(_pulsation*aggr + _phase) * (_pulsation*aggr + _phase) - sin(_pulsation*aggr + _phase)) / std::pow(_pulsation*aggr + _phase, 2);
    _counter++;
}


void omnilearn::Psinc::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _pulsationGradient /= static_cast<double>(_counter);
    _phaseGradient /= static_cast<double>(_counter);

    optimizedUpdate(_pulsation, _previousPulsationGrad, _previousPulsationGrad2, _optimalPreviousPulsationGrad2, _previousPulsationUpdate, _pulsationGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_phase, _previousPhaseGrad, _previousPhaseGrad2, _optimalPreviousPhaseGrad2, _previousPhaseUpdate, _phaseGradient, automaticLearningRate, adaptiveLearningRate, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    _pulsationGradient = 0;
    _phaseGradient = 0;
    _counter = 0;
}


void omnilearn::Psinc::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Sin/Psin activation functions need 2 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _pulsation = coefs[0];
    _phase = coefs[1];
    _pulsationGradient = 0;
    _previousPulsationGrad = 0;
    _previousPulsationGrad2 = 0;
    _optimalPreviousPulsationGrad2 = 0;
    _previousPulsationUpdate = 0;
    _phaseGradient = 0;
    _previousPhaseGrad = 0;
    _previousPhaseGrad2 = 0;
    _optimalPreviousPhaseGrad2 = 0;
    _previousPhaseUpdate = 0;
    _counter = 0;
}


omnilearn::Activation omnilearn::Psinc::signature() const
{
    return Activation::Psinc;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== SOFTMAX FUNCTION ========================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Vector omnilearn::singleSoftmax(Vector input)
{
    double c = input.maxCoeff();
    double sum = 0;
    //subtraction for stability
    for(eigen_size_t j = 0; j < input.size(); j++)
    {
        sum += std::exp(input(j) - c);
    }
    for(eigen_size_t j = 0; j < input.size(); j++)
    {
        input(j) = std::exp(input(j) - c) / sum;
    }
    return input;
}


omnilearn::Matrix omnilearn::softmax(Matrix inputs)
{
    for(eigen_size_t i = 0; i < inputs.rows(); i++)
    {
        double c = inputs.row(i).maxCoeff();
        double sum = 0;
        //subtraction for stability
        for(eigen_size_t j = 0; j < inputs.cols(); j++)
        {
            sum += std::exp(inputs(i,j) - c);
        }
        for(eigen_size_t j = 0; j < inputs.cols(); j++)
        {
            inputs(i,j) = std::exp(inputs(i,j) - c) / sum;
        }
    }
    return inputs;
}