// Activation.hh

#ifndef OMNILEARN_ACTIVATION_HH_
#define OMNILEARN_ACTIVATION_HH_

#include <map>
#include <memory>

#include "Exception.hh"
#include "Matrix.hh"



namespace omnilearn
{



namespace Activation
{
static const size_t Linear = 0;
static const size_t Sigmoid = 1;
static const size_t Tanh = 2;
static const size_t Softplus = 3;
static const size_t Relu = 4;
static const size_t Prelu = 5;
static const size_t Elu = 6;
static const size_t Pelu = 7;
static const size_t Srelu = 8;
static const size_t Gauss = 9;
static const size_t Softexp = 10;
static const size_t Psoftexp = 11;
} // namespace Activation



// interface
class ActivationFct
{
public:
    virtual ~ActivationFct(){}
    virtual double activate(double val) const = 0;
    virtual double prime(double val) const = 0;
    virtual void learn(double gradient, double learningRate) = 0;
    virtual void setCoefs(Vector const& coefs) = 0;
    virtual rowVector getCoefs() const = 0;
    virtual size_t id() const = 0;
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
* id 9  : gauss
* id 10 : softexp
* id 11 : psoftexp
*/



class Linear : public ActivationFct
{
public:
    Linear(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();
};



class Sigmoid : public ActivationFct
{
public:
    Sigmoid(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();
};



class  Tanh : public ActivationFct
{
public:
    Tanh(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();
};



class Softplus : public ActivationFct
{
public:
    Softplus(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();
};



class Relu : public ActivationFct
{
public:
    Relu(Vector const& coefs = (Vector(1) << 0.01).finished());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();

protected:
    double _coef;
    double _savedCoef;
};



class Prelu : public Relu
{
public:
    Prelu(Vector const& coefs = (Vector(1) << 0.01).finished());
    void learn(double gradient, double learningRate);
    size_t id() const;
};



class Elu : public ActivationFct
{
public:
    Elu(Vector const& coefs = (Vector(1) << 0.01).finished());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();

protected:
    double _coef;
    double _savedCoef;
};



class Pelu : public Elu
{
public:
    Pelu(Vector const& coefs = (Vector(1) << 0.01).finished());
    void learn(double gradient, double learningRate);
    size_t id() const;
};



class Srelu : public ActivationFct
{
public:
    Srelu(Vector const& coefs = (Vector(5) << 1.0, 0.1, 1.0, -1.0, 1.0).finished());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();

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



class Gauss : public ActivationFct
{
public:
    //Gauss(); // should take mean and deviation, and make a parametric version
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();
};



// IMPLEMENT SOFTEXP HERE



class Psoftexp : public ActivationFct
{
public:
    Psoftexp(Vector const& coefs = (Vector(1) << 0.01).finished());
    double activate(double val) const;
    double prime(double val) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();

protected:
    double _coef;
    double _savedCoef;
};



Vector singleSoftmax(Vector input);
Matrix softmax(Matrix inputs);


static std::map<size_t, std::function<std::shared_ptr<ActivationFct>()>> activationMap = {
    {0, []{return std::make_shared<Linear>();}},
    {1, []{return std::make_shared<Sigmoid>();}},
    {2, []{return std::make_shared<Tanh>();}},
    {3, []{return std::make_shared<Softplus>();}},
    {4, []{return std::make_shared<Relu>();}},
    {5, []{return std::make_shared<Prelu>();}},
    {6, []{return std::make_shared<Elu>();}},
    {7, []{return std::make_shared<Pelu>();}},
    {8, []{return std::make_shared<Srelu>();}},
    {9, []{return std::make_shared<Gauss>();}},
    {11, []{return std::make_shared<Psoftexp>();}}
};



} //namespace omnilearn



#endif //OMNILEARN_ACTIVATION_HH_