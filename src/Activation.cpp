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


void omnilearn::Linear::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Linear::setCoefs([[maybe_unused]] Vector const& coefs)
{
    //nothing to do
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
    return val * (1 - val);
}


void omnilearn::Sigmoid::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Sigmoid::setCoefs([[maybe_unused]] Vector const& coefs)
{
    //nothing to do
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
    return -1/std::pow(std::cosh(val),2);
}


void omnilearn::Tanh::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Tanh::setCoefs([[maybe_unused]] Vector const& coefs)
{
    //nothing to do
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


void omnilearn::Softplus::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Softplus::setCoefs([[maybe_unused]] Vector const& coefs)
{
    //nothing to do
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


void omnilearn::Relu::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Relu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Relu/Prelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
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
//=== PARAMETRIC RELU ACTIVATION ==============================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Prelu::Prelu(Vector const& coefs) : Relu(coefs)
{
}


void omnilearn::Prelu::learn(double gradient, double learningRate)
{
    //TO BE IMPLEMENTED
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


void omnilearn::Elu::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Elu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Elu/Pelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _coef = coefs[0];
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



omnilearn::Pelu::Pelu(Vector const& coefs) : Elu(coefs)
{
}


void omnilearn::Pelu::learn(double gradient, double learningRate)
{
    //TO BE IMPLEMENTED
}


omnilearn::Activation omnilearn::Pelu::signature() const
{
    return Activation::Pelu;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== S-SHAPED ACTIVATION =====================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Srelu::Srelu(Vector const& coefs)
{
    if(coefs.size() != 5)
        // 3 coefs and 2 hinges
        throw Exception("Srelu activation function needs 5 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _coef1 = coefs[0];
    _coef2 = coefs[1];
    _coef3 = coefs[2];
    _hinge1 = coefs[3];
    _hinge2 = coefs[4];
}


double omnilearn::Srelu::activate(double val) const
{
    // TO BE IMPLEMENTED
    return 0;
}


double omnilearn::Srelu::prime(double val) const
{
    // TO BE IMPLEMENTED
    return 0;
}


void omnilearn::Srelu::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Srelu::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 5)
        // 3 coefs and 2 hinges
        throw Exception("Srelu activation function needs 5 coefficients. " + std::to_string(coefs.size()) + " provided.");
    _coef1 = coefs[0];
    _coef2 = coefs[1];
    _coef3 = coefs[2];
    _hinge1 = coefs[3];
    _hinge2 = coefs[4];

    _savedCoef1 = 0;
    _savedCoef2 = 0;
    _savedCoef3 = 0;
    _savedHinge1 = 0;
    _savedHinge2 = 0;
}


omnilearn::rowVector omnilearn::Srelu::getCoefs() const
{
    return (Vector(5) << _coef1, _coef2, _coef3, _hinge1, _hinge2).finished();
}


omnilearn::Activation omnilearn::Srelu::signature() const
{
    return Activation::Srelu;
}


void omnilearn::Srelu::keep()
{
    _savedCoef1 = _coef1;
    _savedCoef2 = _coef2;
    _savedCoef3 = _coef3;
    _savedHinge1 = _hinge1;
    _savedHinge2 = _hinge2;
}


void omnilearn::Srelu::release()
{
    _coef1 = _savedCoef1;
    _coef2 = _savedCoef2;
    _coef3 = _savedCoef3;
    _hinge1 = _savedHinge1;
    _hinge2 = _savedHinge2;
}




//=============================================================================
//=============================================================================
//=============================================================================
//=== GAUSSIAN ACTIVATION =====================================================
//=============================================================================
//=============================================================================
//=============================================================================



double omnilearn::Gauss::activate(double val) const
{
    return std::exp(-std::pow(val, 2));
}


double omnilearn::Gauss::prime(double val) const
{
    return -2 * val * std::exp(-std::pow(val, 2));
}


void omnilearn::Gauss::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Gauss::setCoefs([[maybe_unused]] Vector const& coefs)
{
    //nothing to do
}


omnilearn::rowVector omnilearn::Gauss::getCoefs() const
{
    return Vector(0);
}


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
        throw Exception("Softexp activation function needs 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
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
        return (_coef < 0 ? 1 / (1 - (_coef * (_coef + val))) : std::exp(_coef * val));
    else
        return std::exp(_coef * val);
}


void omnilearn::Softexp::learn(double gradient, double learningRate)
{
    //TO BE IMPLEMENTED
}


void omnilearn::Softexp::setCoefs([[maybe_unused]] Vector const& coefs)
{
    //nothing to do
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
//=== PARAMETRIC SOFTEXP ACTIVATION ===========================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Psoftexp::Psoftexp(Vector const& coefs):
Softexp(coefs)
{
}


void omnilearn::Psoftexp::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //TO BE IMPLEMENTED
}


omnilearn::Activation omnilearn::Psoftexp::signature() const
{
    return Activation::Psoftexp;
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