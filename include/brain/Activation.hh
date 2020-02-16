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
    virtual void setCoefs(Vector const& coefs) = 0;
    virtual rowVector getCoefs() const = 0;
    virtual void save() = 0;
    virtual void loadSaved() = 0;
};


/*
* id 0  : linear
* id 1  : sigmoid
* id 2  : tanh
* id 3  : softplus
* id 4  : relu
* id 5  : prelu
* id 6  : elu
* id 7  : pelu
* id 8  : srelu
* id 9  : gaussian
* id 10 : softexp
*/


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

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        //nothing to do
    }

    rowVector getCoefs() const
    {
        return Vector(0);
    }

    static size_t id()
    {
        return 0;
    }

    void save()
    {
        //nothing to do
    }

    void loadSaved()
    {
        //nothing to do
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

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        //nothing to do
    }

    rowVector getCoefs() const
    {
        return Vector(0);
    }

    static size_t id()
    {
        return 1;
    }

    void save()
    {
        //nothing to do
    }

    void loadSaved()
    {
        //nothing to do
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

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        //nothing to do
    }

    rowVector getCoefs() const
    {
        return Vector(0);
    }

    static size_t id()
    {
        return 2;
    }

    void save()
    {
        //nothing to do
    }

    void loadSaved()
    {
        //nothing to do
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

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        //nothing to do
    }

    rowVector getCoefs() const
    {
        return Vector(0);
    }

    static size_t id()
    {
        return 3;
    }

    void save()
    {
        //nothing to do
    }

    void loadSaved()
    {
        //nothing to do
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
        _savedCoef = 0;
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

    void setCoefs(Vector const& coefs)
    {
        if(coefs.size() != 1)
            throw Exception("Relu/Prelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
        _coef = coefs[0];
    }

    rowVector getCoefs() const
    {
        return (Vector(1) << _coef).finished();
    }

    static size_t id()
    {
        return 4;
    }

    void save()
    {
        _savedCoef = _coef;
    }

    void loadSaved()
    {
        _coef = _savedCoef;
    }

protected:
    double _coef;
    double _savedCoef;
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

    static size_t id()
    {
        return 5;
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
        _savedCoef = 0;
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

    void setCoefs(Vector const& coefs)
    {
        if(coefs.size() != 1)
            throw Exception("Elu/Pelu activation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
        _coef = coefs[0];
    }

    rowVector getCoefs() const
    {
        return (Vector(1) << _coef).finished();
    }

    static size_t id()
    {
        return 6;
    }

    void save()
    {
        _savedCoef = _coef;
    }

    void loadSaved()
    {
        _coef = _savedCoef;
    }

protected:
    double _coef;
    double _savedCoef;
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

    static size_t id()
    {
        return 7;
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

    void setCoefs(Vector const& coefs)
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

    rowVector getCoefs() const
    {
        (Vector(5) << _coef1, _coef2, _coef3, _hinge1, _hinge2).finished();
    }

    static size_t id()
    {
        return 8;
    }

    void save()
    {
        _savedCoef1 = _coef1;
        _savedCoef2 = _coef2;
        _savedCoef3 = _coef3;
        _savedHinge1 = _hinge1;
        _savedHinge2 = _hinge2;
    }

    void loadSaved()
    {
        _coef1 = _savedCoef1;
        _coef2 = _savedCoef2;
        _coef3 = _savedCoef3;
        _hinge1 = _savedHinge1;
        _hinge2 = _savedHinge2;
    }

protected:
    double _coef1;
    double _coef2;
    double _coef3;
    double _hinge1;
    double _hinge2;

    double _savedCoef1;
    double _savedCoef2;
    double _savedCoef3;
    double _savedHinge1;
    double _savedHinge2;
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

    rowVector getCoefs() const
    {
        return Vector(0);
    }

    static size_t id()
    {
        return 9;
    }

    void save()
    {
        //nothing to do
    }

    void loadSaved()
    {
        //nothing to do
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

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        //nothing to do
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