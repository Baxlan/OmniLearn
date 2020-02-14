#ifndef BRAIN_ACTIVATION_HH_
#define BRAIN_ACTIVATION_HH_

#include "Matrix.hh"
#include "Exception.hh"

namespace brain
{



//interface
class Activation
{
public:
    virtual ~Activation(){}
    virtual double activate(double val) const = 0;
    virtual double prime(double val) const = 0;
    virtual void learn(double gradient, double learningRate) = 0;
    virtual Vector getCoefs() const = 0;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== LINEAR ACTIVATION =======================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Linear : public Activation
{
public:
    Linear(Vector const& coefs = Vector())
    {
        if(coefs.size() != 0)
            throw Exception("Linear activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
    }

    double activate(double val) const
    {
        return val;
    }

    double prime([[maybe_unused]] double val) const
    {
        return 1;
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        return Vector(0);
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== SIGMOID ACTIVATION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Sigmoid : public Activation
{
public:
    Sigmoid(Vector const& coefs = Vector())
    {
        if(coefs.size() != 0)
            throw Exception("Sigmoid activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
    }

    double activate(double val) const
    {
        return 1 / (1 + std::exp(-val));
    }

    double prime(double val) const
    {
        return val * (1 - val);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        return Vector(0);
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== TANH ACTIVATION =========================================================
//=============================================================================
//=============================================================================
//=============================================================================



class  Tanh : public Activation
{
public:
    Tanh(Vector const& coefs = Vector())
    {
        if(coefs.size() != 0)
            throw Exception("Tanh activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
    }

    double activate(double val) const
    {
        return std::tanh(val);
    }

    double prime(double val) const
    {
        return -1/std::pow(std::cosh(val),2);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        return Vector(0);
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== SOFTPLUS ACTIVATION =====================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Softplus : public Activation
{
public:
    Softplus(Vector const& coefs = Vector())
    {
        if(coefs.size() != 0)
            throw Exception("Softplus activation function doesn't need coefficients. " + std::to_string(coefs.size()) + " provided.");
    }

    double activate(double val) const
    {
        return std::log(std::exp(val) + 1);
    }

    double prime(double val) const
    {
        return 1 / (1 + std::exp(-val));
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        return Vector(0);
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== (LEAKY) RELU ACTIVATION =================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Relu : public Activation
{
public:
    Relu(Vector const& coefs = (Vector(1) << 0.01).finished())
    {
        if(coefs.size() != 1)
            throw Exception("Relu/Prelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
        _coef = coefs[0];
    }

    double activate(double val) const
    {
        return (val < 0 ? _coef*val : val);
    }

    double prime(double val) const
    {
        return (val < 0 ? _coef : 1);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        return (Vector(1) << _coef).finished();
    }

protected:
    double _coef;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC RELU ACTIVATION ==============================================
//=============================================================================
//=============================================================================
//=============================================================================



class Prelu : public Relu
{
public:
    Prelu(Vector const& coefs = (Vector(1) << 0.01).finished()) : Relu(coefs)
    {
    }

    void learn(double gradient, double learningRate)
    {
        //TO BE IMPLEMENTED
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== EXPONENTIAL RELU ACTIVATION =============================================
//=============================================================================
//=============================================================================
//=============================================================================



class Elu : public Activation
{
public:
    Elu(Vector const& coefs = (Vector(1) << 0.01).finished())
    {
        if(coefs.size() != 1)
            throw Exception("Elu/Pelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
        _coef = coefs[0];
    }

    double activate(double val) const
    {
        return (val < 0 ? _coef*(std::exp(val)-1) : val);
    }

    double prime(double val) const
    {
        return (val < 0 ? _coef * std::exp(val) : 1);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        return (Vector(1) << _coef).finished();
    }

protected:
    double _coef;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC EXPONENTIAL ACTIVATION =======================================
//=============================================================================
//=============================================================================
//=============================================================================



class Pelu : public Elu
{
public:
    Pelu(Vector const& coefs = (Vector(1) << 0.01).finished()) : Elu(coefs)
    {
    }

    void learn(double gradient, double learningRate)
    {
        //TO BE IMPLEMENTED
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== S-SHAPED ACTIVATION =====================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Srelu : public Activation
{
public:
    Srelu(Vector const& coefs = (Vector(5) << 1.0, 0.1, 1.0, -1.0, 1.0).finished())
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

    double activate(double val) const
    {

    }

    double prime(double val) const
    {

    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        (Vector(5) << _coef1, _coef2, _coef3, _hinge1, _hinge2).finished();
    }

protected:
    double _coef1;
    double _coef2;
    double _coef3;
    double _hinge1;
    double _hinge2;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== GAUSSIAN ACTIVATION =====================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Gauss : public Activation
{
public:
    double activate(double val) const
    {
        return std::exp(-std::pow(val, 2));
    }

    double prime(double val) const
    {
        return -2 * val * std::exp(-std::pow(val, 2));
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    Vector getCoefs() const
    {
        return Vector(0);
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== SOFTEXP ACTIVATION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



// TO BE IMPLEMENTED



//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC SOFTEXP ACTIVATION ===========================================
//=============================================================================
//=============================================================================
//=============================================================================



class Softexp : public Activation
{
public:
    double activate(double val) const
    {
        if(_coef < -std::numeric_limits<double>::epsilon())
            return -std::log(1-(_coef*(val + _coef))) / _coef;
        else if(_coef > std::numeric_limits<double>::epsilon())
            return ((std::exp(_coef * val) - 1) / _coef) + _coef;
        else
            return val;
    }

    double prime(double val) const
    {
        if(_coef < 0)
            return (_coef < 0 ? 1 / (1 - (_coef * (_coef + val))) : std::exp(_coef * val));
        else
            return std::exp(_coef * val);
    }

    void learn(double gradient, double learningRate)
    {
        //TO BE IMPLEMENTED
    }

    Vector getCoefs()
    {
        return Vector(0);
    }

protected:
    double _coef;
};


//=============================================================================
//=============================================================================
//=============================================================================
//=== SOFTMAX FUNCTION ========================================================
//=============================================================================
//=============================================================================
//=============================================================================


Vector singleSoftmax(Vector input)
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


Matrix softmax(Matrix inputs)
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

} //namespace brain



#endif //BRAIN_ACTIVATION_HH_