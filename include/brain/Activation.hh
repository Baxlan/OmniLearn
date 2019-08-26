#ifndef BRAIN_ACTIVATION_HH_
#define BRAIN_ACTIVATION_HH_

#include <cmath>

namespace brain
{



//interface
class Activation
{
public:
    virtual ~Activation(){}
    virtual double activate(double val) const = 0;
    virtual double prime(double val) const = 0;
    virtual void learn(double gradient, double learningRate, double momentum) = 0;
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
    double activate(double val) const
    {
        return val;
    }

    double prime([[maybe_unused]] double val) const
    {
        return 1;
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
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
    double activate(double val) const
    {
        return 1 / (1 + std::exp(-val));
    }

    double prime(double val) const
    {
        return val * (1 - val);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
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
    double activate(double val) const
    {
        return std::tanh(val);
    }

    double prime(double val) const
    {
        return -1/std::pow(std::cosh(val),2);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
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
    double activate(double val) const
    {
        return std::log(std::exp(val) + 1);
    }

    double prime(double val) const
    {
        return 1 / (1 + std::exp(-val));
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
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
    Relu(double coef = 0.01) : _coef(coef)
    {
    }

    double activate(double val) const
    {
        return (val < 0 ? _coef*val : val);
    }

    double prime(double val) const
    {
        return (val < 0 ? _coef : 1);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
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



class Prelu : public Activation
{
public:
    Prelu(double coef = 0) : _coef(coef)
    {
    }

    double activate(double val) const
    {
        return (val < 0 ? _coef*val : val);
    }

    double prime(double val) const
    {
        return (val < 0 ? _coef : 1);
    }

    void learn(double gradient, double learningRate, double momentum)
    {
        //TO BE IMPLEMENTED
    }

protected:
    double _coef;
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
    Elu(double coef = 0) : _coef(coef)
    {
    }

    double activate(double val) const
    {
        return (val < 0 ? _coef*(std::exp(val)-1) : val);
    }

    double prime(double val) const
    {
        return (val < 0 ? _coef * std::exp(val) : 1);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
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



class Pelu : public Activation
{
public:
    Pelu(double coef = 0) : _coef(coef)
    {
    }

    double activate(double val) const
    {
        return (val < 0 ? _coef*(std::exp(val)-1) : val);
    }

    double prime(double val) const
    {
        return (val < 0 ? _coef * std::exp(val) : 1);
    }

    void learn(double gradient, double learningRate, double momentum)
    {
        //TO BE IMPLEMENTED
    }

protected:
    double _coef;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== S-SHAPED ACTIVATION =====================================================
//=============================================================================
//=============================================================================
//=============================================================================



// if there are two hinges, then this is S-shaped rectified linear unit (Srelu)
class Srelu : public Activation
{
public:
    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }

protected:

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

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
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
        if(_coef < 0)
            return -std::log(1-(_coef*(val + _coef))) / _coef;
        if(_coef == 0)
            return val;
        if(_coef > 0)
            return ((std::exp(_coef * val) - 1) / _coef) + _coef;
    }

    double prime(double val) const
    {
        if(_coef < 0)
            return (_coef < 0 ? 1 / (1 - (_coef * (_coef + val))) : std::exp(_coef * val));
        else
            return std::exp(_coef * val);
    }

    void learn(double gradient, double learningRate, double momentum)
    {
        //TO BE IMPLEMENTED
    }
protected:
    double _coef;
};



} //namespace brain



#endif //BRAIN_ACTIVATION_HH_