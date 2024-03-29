// Activation.cpp

#include "omnilearn/Activation.hh"
#include "omnilearn/Exception.hh"



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


void omnilearn::Linear::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
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


size_t omnilearn::Linear::getNbParameters() const
{
    return 0;
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


void omnilearn::Sigmoid::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
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

//static function
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


size_t omnilearn::Sigmoid::getNbParameters() const
{
    return 0;
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


void omnilearn::Tanh::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
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

//static function
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


size_t omnilearn::Tanh::getNbParameters() const
{
    return 0;
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


void omnilearn::Softplus::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
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

//static function
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


size_t omnilearn::Softplus::getNbParameters() const
{
    return 0;
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


void omnilearn::Relu::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
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

//static function
omnilearn::Activation omnilearn::Relu::signature() const
{
    return Activation::Relu;
}


void omnilearn::Relu::keep()
{
    //nothing to do
}


void omnilearn::Relu::release()
{
    //nothing to do
}


size_t omnilearn::Relu::getNbParameters() const
{
    return 0;
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
_coefInfos(),
_counter(0)
{
}


void omnilearn::Prelu::computeGradients(double aggr, double inputGrad)
{
    _coefInfos.gradient -= inputGrad * (aggr < 0 ? aggr : 0);
    _counter += 1;
}


void omnilearn::Prelu::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _coefInfos.gradient /= static_cast<double>(_counter);
    optimizedUpdate(_coef, _coefInfos, automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);

    _coefInfos = LearnableParameterInfos();
    _counter = 0;
}


void omnilearn::Prelu::setCoefs(Vector const& coefs)
{
    Relu::setCoefs(coefs);

    _coefInfos = LearnableParameterInfos();
    _counter = 0;
}

//static function
omnilearn::Activation omnilearn::Prelu::signature() const
{
    return Activation::Prelu;
}


void omnilearn::Prelu::keep()
{
    _savedCoef = _coef;
}


void omnilearn::Prelu::release()
{
    _coef = _savedCoef;
}


size_t omnilearn::Prelu::getNbParameters() const
{
    return 1;
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
    if(coefs.size() != 2)
        throw Exception("Elu/Pelu activation functions need 2 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _coef2 = coefs[1];
    _savedCoef = 0;
    _savedCoef2 = 0;
}


double omnilearn::Elu::activate(double val) const
{
    return (val < 0 ? _coef*(std::exp(val/_coef2)-1) : (_coef/_coef2) * val);
}


double omnilearn::Elu::prime(double val) const
{
    return (val < 0 ? (_coef * std::exp(val/_coef2) / _coef2) : (_coef/_coef2));
}


void omnilearn::Elu::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Elu::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Elu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 2)
        throw Exception("Elu/Pelu activation functions need 2 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _coef2 = coefs[1];
    _savedCoef = 0;
    _savedCoef2 = 0;
}


omnilearn::rowVector omnilearn::Elu::getCoefs() const
{
    return (Vector(2) << _coef, _coef2).finished();
}

//static function
omnilearn::Activation omnilearn::Elu::signature() const
{
    return Activation::Elu;
}


void omnilearn::Elu::keep()
{
    //nothing to do
}


void omnilearn::Elu::release()
{
    //nothing to do
}


size_t omnilearn::Elu::getNbParameters() const
{
    return 0;
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
_coef1Infos(),
_coef2Infos(),
_counter(0)
{
}


void omnilearn::Pelu::computeGradients(double aggr, double inputGrad)
{
    _coef1Infos.gradient += inputGrad * (aggr < 0 ? std::exp(aggr/_coef2)-1 : aggr/_coef2);
    _coef2Infos.gradient += inputGrad * (aggr < 0 ? -aggr*_coef*std::exp(aggr/_coef2)/std::pow(_coef2, 2) : -_coef*aggr/std::pow(_coef2,2));
    _counter += 1;
}


void omnilearn::Pelu::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _coef1Infos.gradient /= static_cast<double>(_counter);
    _coef2Infos.gradient /= static_cast<double>(_counter);

    optimizedUpdate(_coef, _coef1Infos, automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_coef2, _coef2Infos, automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay, true);

    _coef1Infos = LearnableParameterInfos();
    _coef2Infos = LearnableParameterInfos();
    _counter = 0;
}


void omnilearn::Pelu::setCoefs(Vector const& coefs)
{
    Elu::setCoefs(coefs);

    _coef1Infos = LearnableParameterInfos();
    _coef2Infos = LearnableParameterInfos();
    _counter = 0;
}

//static function
omnilearn::Activation omnilearn::Pelu::signature() const
{
    return Activation::Pelu;
}


void omnilearn::Pelu::keep()
{
    _savedCoef = _coef;
    _savedCoef2 = _coef2;
}


void omnilearn::Pelu::release()
{
    _coef = _savedCoef;
    _coef2 = _savedCoef2;
}


size_t omnilearn::Pelu::getNbParameters() const
{
    return 2;
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


void omnilearn::Gauss::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
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

//static function
omnilearn::Activation omnilearn::Gauss::signature() const
{
    return Activation::Gauss;
}


void omnilearn::Gauss::keep()
{
    //nothing to do
}


void omnilearn::Gauss::release()
{
    //nothing to do
}


size_t omnilearn::Gauss::getNbParameters() const
{
    return 0;
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
_centerInfos(),
_devInfos(),
_counter(0)
{
}


void omnilearn::Pgauss::computeGradients(double aggr, double inputGrad)
{
    _centerInfos.gradient += inputGrad * (activate(aggr) * (aggr - _center) / std::pow(_dev, 2));
    _devInfos.gradient += inputGrad * (activate(aggr) * std::pow((aggr - _center), 2) / std::pow(_dev, 3));
    _counter += 1;
}


void omnilearn::Pgauss::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _centerInfos.gradient /= static_cast<double>(_counter);
    _devInfos.gradient /= static_cast<double>(_counter);

    optimizedUpdate(_center, _centerInfos, automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay);
    optimizedUpdate(_dev, _devInfos, automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay, true);

    _devInfos = LearnableParameterInfos();
    _devInfos = LearnableParameterInfos();
    _counter = 0;
}


void omnilearn::Pgauss::setCoefs(Vector const& coefs)
{
    Gauss::setCoefs(coefs);

    _devInfos = LearnableParameterInfos();
    _devInfos = LearnableParameterInfos();
    _counter = 0;
}

//static function
omnilearn::Activation omnilearn::Pgauss::signature() const
{
    return Activation::Pgauss;
}


void omnilearn::Pgauss::keep()
{
    _savedCenter = _center;
    _savedDev = _dev;
}


void omnilearn::Pgauss::release()
{
    _center = _savedCenter;
    _dev = _savedDev;
}


size_t omnilearn::Pgauss::getNbParameters() const
{
    return 2;
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