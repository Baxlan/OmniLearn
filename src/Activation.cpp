// Activation.cpp

#include "omnilearn/Activation.hh"
#include "omnilearn/Exception.hh"


#include <iostream>
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


void omnilearn::Linear::updateCoefs([[maybe_unused]] double learningRate)
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


void omnilearn::Linear::nesterov()
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


void omnilearn::Sigmoid::updateCoefs([[maybe_unused]] double learningRate)
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


void omnilearn::Sigmoid::nesterov()
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


void omnilearn::Tanh::updateCoefs([[maybe_unused]] double learningRate)
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


void omnilearn::Tanh::nesterov()
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


void omnilearn::Softplus::updateCoefs([[maybe_unused]] double learningRate)
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


void omnilearn::Softplus::nesterov()
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


void omnilearn::Relu::updateCoefs([[maybe_unused]] double learningRate)
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


void omnilearn::Relu::nesterov()
{
    //nothing to do
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
_counter(0)
{
}


void omnilearn::Prelu::computeGradients(double aggr, double inputGrad)
{
    _coefGradient += (aggr < 0 ? aggr * inputGrad : 0);
    _counter += 1;
}


void omnilearn::Prelu::updateCoefs(double learningRate)
{
    _coefGradient /= static_cast<double>(_counter);

    _coef += learningRate * _coefGradient;

    _coefGradient = 0;
    _counter = 0;
}


omnilearn::Activation omnilearn::Prelu::signature() const
{
    return Activation::Prelu;
}


void omnilearn::Prelu::nesterov()
{
    //nothing to do
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


void omnilearn::Elu::updateCoefs([[maybe_unused]] double learningRate)
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


void omnilearn::Elu::nesterov()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC EXPONENTIAL RELU ACTIVATION ==================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Pelu::Pelu(Vector const& coefs):
Elu(coefs)
{
}


void omnilearn::Pelu::computeGradients(double aggr, double inputGrad)
{

}


void omnilearn::Pelu::updateCoefs(double learningRate)
{

}


omnilearn::Activation omnilearn::Pelu::signature() const
{
    return Activation::Pelu;
}


void omnilearn::Pelu::nesterov()
{
    //nothing to do
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

}


void omnilearn::Srelu::updateCoefs(double learningRate)
{

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


void omnilearn::Srelu::nesterov()
{
    //nothing to do
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
    if(coefs.size() != 3)
        throw Exception("Gauss/Pgauss activation functions need 3 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _center = coefs[0];
    _dev = coefs[1];
    _coef = coefs[2];

    _savedCenter = 0;
    _savedDev = 0;
    _savedCoef = 0;
}


double omnilearn::Gauss::activate(double val) const
{
    return _coef * std::exp(-std::pow(val - _center, 2) / (2*std::pow(_dev, 2)));
}


double omnilearn::Gauss::prime(double val) const
{
    return activate(val) * (_center - val) / std::pow(_dev, 2);
}


void omnilearn::Gauss::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Gauss::updateCoefs([[maybe_unused]] double learningRate)
{
    //nothing to do
}


void omnilearn::Gauss::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 3)
        throw Exception("Gauss/Pgauss activation functions need 3 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _center = coefs[0];
    _dev = coefs[1];
    _coef = coefs[2];

    _savedCenter = 0;
    _savedDev = 0;
    _savedCoef = 0;
}


omnilearn::rowVector omnilearn::Gauss::getCoefs() const
{
    return (Vector(3) << _center, _dev, _coef).finished();
}


omnilearn::Activation omnilearn::Gauss::signature() const
{
    return Activation::Gauss;
}


void omnilearn::Gauss::keep()
{
    _savedCenter = _center;
    _savedDev = _dev;
    _savedCoef = _coef;
}


void omnilearn::Gauss::release()
{
    _center = _savedCenter;
    _dev = _savedDev;
    _coef = _savedCoef;
}


void omnilearn::Gauss::nesterov()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC GAUSS ACTIVATION =============================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Pgauss::Pgauss(Vector const& coefs):
Gauss(coefs)
{
}


void omnilearn::Pgauss::computeGradients(double aggr, double inputGrad)
{

}


void omnilearn::Pgauss::updateCoefs(double learningRate)
{

}


omnilearn::Activation omnilearn::Pgauss::signature() const
{
    return Activation::Pgauss;
}


void omnilearn::Pgauss::nesterov()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== SOFTEXP ACTIVATION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Softexp::Softexp(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Softexp/Psoftexp activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
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


void omnilearn::Softexp::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Softexp::updateCoefs([[maybe_unused]] double learningRate)
{
    //nothing to do
}


void omnilearn::Softexp::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Softexp/Psoftexp activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
    _savedCoef = 0;
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


void omnilearn::Softexp::nesterov()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC SOFTEXP ACTIVATION ===========================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Psoftexp::Psoftexp(Vector const& coefs):
Softexp(coefs)
{
}


void omnilearn::Psoftexp::computeGradients(double aggr, double inputGrad)
{

}


void omnilearn::Psoftexp::updateCoefs(double learningRate)
{

}


omnilearn::Activation omnilearn::Psoftexp::signature() const
{
    return Activation::Psoftexp;
}


void omnilearn::Psoftexp::nesterov()
{
    //nothing to do
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
    if(coefs.size() != 0)
        throw Exception("Sin activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


double omnilearn::Sin::activate(double val) const
{
    return std::sin(val);
}


double omnilearn::Sin::prime([[maybe_unused]] double val) const
{
    return std::cos(val);
}


void omnilearn::Sin::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Sin::updateCoefs([[maybe_unused]] double learningRate)
{
    //nothing to do
}


void omnilearn::Sin::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Sin activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Sin::getCoefs() const
{
    return Vector(0);
}

//static function
omnilearn::Activation omnilearn::Sin::signature() const
{
    return Activation::Sin;
}


void omnilearn::Sin::keep()
{
    //nothing to do
}


void omnilearn::Sin::release()
{
    //nothing to do
}


void omnilearn::Sin::nesterov()
{
    //nothing to do
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
    if(coefs.size() != 0)
        throw Exception("Sinc activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


double omnilearn::Sinc::activate(double val) const
{
    if(val > -std::numeric_limits<double>::epsilon() && val < std::numeric_limits<double>::epsilon())
        return 1;
    else
        return std::sin(val)/val;
}


double omnilearn::Sinc::prime([[maybe_unused]] double val) const
{
    if(val > -std::numeric_limits<double>::epsilon() && val < std::numeric_limits<double>::epsilon())
        return 0;
    else
        return (std::cos(val)/val) - (std::sin(val)/std::pow(val, 2));
}


void omnilearn::Sinc::computeGradients([[maybe_unused]] double aggr, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Sinc::updateCoefs([[maybe_unused]] double learningRate)
{
    //nothing to do
}


void omnilearn::Sinc::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Sinc activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Sinc::getCoefs() const
{
    return Vector(0);
}

//static function
omnilearn::Activation omnilearn::Sinc::signature() const
{
    return Activation::Sinc;
}


void omnilearn::Sinc::keep()
{
    //nothing to do
}


void omnilearn::Sinc::release()
{
    //nothing to do
}


void omnilearn::Sinc::nesterov()
{
    //nothing to do
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