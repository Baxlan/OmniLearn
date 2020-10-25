// Activation.hh

#ifndef OMNILEARN_ACTIVATION_HH_
#define OMNILEARN_ACTIVATION_HH_

#include <map>
#include <memory>
#include <string>

#include "Matrix.hh"



namespace omnilearn
{



enum class Activation {Linear, Sigmoid, Tanh, Softplus, Relu, Prelu, Elu, Pelu, Srelu, Gauss, Pgauss, Sin, Psin, Sinc, Psinc};



// interface
class IActivation
{
public:
    virtual ~IActivation(){}
    virtual double activate(double val) const = 0;
    virtual double prime(double val) const = 0;
    virtual void computeGradients(double aggr, double inputGrad) = 0;
    virtual void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay) = 0;
    virtual void setCoefs(Vector const& coefs) = 0;
    virtual rowVector getCoefs() const = 0;
    virtual Activation signature() const = 0;
    virtual void keep() = 0;
    virtual void release() = 0;
};



class Linear : public IActivation
{
public:
    Linear(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();
};



class Sigmoid : public IActivation
{
public:
    Sigmoid(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();
};



class  Tanh : public IActivation
{
public:
    Tanh(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();
};



class Softplus : public IActivation
{
public:
    Softplus(Vector const& coefs = Vector());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();
};



class Relu : public IActivation
{
public:
    Relu(Vector const& coefs = (Vector(1) << 0.01).finished());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();

protected:
    double _coef;
    double _savedCoef;
};



class Prelu : public Relu
{
public:
    Prelu(Vector const& coefs = (Vector(1) << 0.01).finished());
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    Activation signature() const;

protected:
    double _coefGradient;
    double _previousCoefGrad;
    double _previousCoefGrad2;
    double _optimalPreviousCoefGrad2;
    double _previousCoefUpdate;
    double _counter;
};



class Elu : public IActivation
{
public:
    Elu(Vector const& coefs = (Vector(2) << 1., 1.).finished());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();

protected:
    double _coef;
    double _coef2;
    double _savedCoef;
    double _savedCoef2;
};



class Pelu : public Elu
{
public:
    Pelu(Vector const& coefs = (Vector(2) << 1., 1.).finished());
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    Activation signature() const;

protected:
    double _coefGradient;
    double _previousCoefGrad;
    double _previousCoefGrad2;
    double _optimalPreviousCoefGrad2;
    double _previousCoefUpdate;
    double _coef2Gradient;
    double _previousCoef2Grad;
    double _previousCoef2Grad2;
    double _optimalPreviousCoef2Grad2;
    double _previousCoef2Update;
    double _counter;
};



class Srelu : public IActivation
{
public:
    Srelu(Vector const& coefs = (Vector(4) << 0.01, 1, 0, 1.0).finished());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();

protected:
    double _coef1;
    double _coef2;
    double _hinge1;
    double _hinge2;

    double _savedCoef1;
    double _savedCoef2;
    double _savedHinge1;
    double _savedHinge2;

    double _coef1Gradient;
    double _coef2Gradient;
    double _hinge1Gradient;
    double _hinge2Gradient;

    double _previousCoef1Grad;
    double _previousCoef2Grad;
    double _previousHinge1Grad;
    double _previousHinge2Grad;

    double _previousCoef1Grad2;
    double _previousCoef2Grad2;
    double _previousHinge1Grad2;
    double _previousHinge2Grad2;

    double _optimalPreviousCoef1Grad2;
    double _optimalPreviousCoef2Grad2;
    double _optimalPreviousHinge1Grad2;
    double _optimalPreviousHinge2Grad2;

    double _previousCoef1Update;
    double _previousCoef2Update;
    double _previousHinge1Update;
    double _previousHinge2Update;

    double _counter1;
    double _counter2;
};



class Gauss : public IActivation
{
public:
    Gauss(Vector const& coefs = (Vector(2) << 0.0, 1.0).finished());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();

protected:
    double _center;
    double _dev;
    double _savedCenter;
    double _savedDev;
};



class Pgauss : public Gauss
{
public:
    Pgauss(Vector const& coefs = (Vector(2) << 0.0, 1.0).finished());
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    Activation signature() const;

protected:
    double _centerGradient;
    double _previousCenterGrad;
    double _previousCenterGrad2;
    double _optimalPreviousCenterGrad2;
    double _previousCenterUpdate;

    double _devGradient;
    double _previousDevGrad;
    double _previousDevGrad2;
    double _optimalPreviousDevGrad2;
    double _previousDevUpdate;

    double _counter;
};



class Softexp : public IActivation
{
public:
    Softexp(Vector const& coefs = (Vector(1) << 0).finished());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();

protected:
    double _coef;
    double _savedCoef;

    double _coefGradient;
    double _previousCoefGrad;
    double _previousCoefGrad2;
    double _optimalPreviousCoefGrad2;
    double _previousCoefUpdate;
    double _counter;
};



class Sin : public IActivation
{
public:
    Sin(Vector const& coefs = (Vector(2) << 1.0, 0.0).finished());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();

    double _pulsation;
    double _phase;
    double _savedPulsation;
    double _savedPhase;
};



class Psin : public Sin
{
public:
    Psin(Vector const& coefs = (Vector(2) << 1.0, 0.0).finished());
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    Activation signature() const;

protected:
    double _pulsationGradient;
    double _previousPulsationGrad;
    double _previousPulsationGrad2;
    double _optimalPreviousPulsationGrad2;
    double _previousPulsationUpdate;

    double _phaseGradient;
    double _previousPhaseGrad;
    double _previousPhaseGrad2;
    double _optimalPreviousPhaseGrad2;
    double _previousPhaseUpdate;

    double _counter;
};



class Sinc : public IActivation
{
public:
    Sinc(Vector const& coefs = (Vector(2) << 1.0, 0.0).finished());
    double activate(double val) const;
    double prime(double val) const;
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Activation signature() const;
    void keep();
    void release();

    double _pulsation;
    double _phase;
    double _savedPulsation;
    double _savedPhase;
};



class Psinc : public Sinc
{
public:
    Psinc(Vector const& coefs = (Vector(2) << 1.0, 0.0).finished());
    void computeGradients(double aggr, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    Activation signature() const;

protected:
    double _pulsationGradient;
    double _previousPulsationGrad;
    double _previousPulsationGrad2;
    double _optimalPreviousPulsationGrad2;
    double _previousPulsationUpdate;

    double _phaseGradient;
    double _previousPhaseGrad;
    double _previousPhaseGrad2;
    double _optimalPreviousPhaseGrad2;
    double _previousPhaseUpdate;

    double _counter;
};



Vector singleSoftmax(Vector input);
Matrix softmax(Matrix inputs);



static std::map<Activation, std::function<std::unique_ptr<IActivation>()>> activationMap = {
    {Activation::Linear, []{return std::make_unique<Linear>();}},
    {Activation::Sigmoid, []{return std::make_unique<Sigmoid>();}},
    {Activation::Tanh, []{return std::make_unique<Tanh>();}},
    {Activation::Softplus, []{return std::make_unique<Softplus>();}},
    {Activation::Relu, []{return std::make_unique<Relu>();}},
    {Activation::Prelu, []{return std::make_unique<Prelu>();}},
    {Activation::Elu, []{return std::make_unique<Elu>();}},
    {Activation::Pelu, []{return std::make_unique<Pelu>();}},
    {Activation::Srelu, []{return std::make_unique<Srelu>();}},
    {Activation::Gauss, []{return std::make_unique<Gauss>();}},
    {Activation::Pgauss, []{return std::make_unique<Pgauss>();}},
    {Activation::Sin, []{return std::make_unique<Sin>();}},
    {Activation::Psin, []{return std::make_unique<Psin>();}},
    {Activation::Sinc, []{return std::make_unique<Sinc>();}},
    {Activation::Psinc, []{return std::make_unique<Psinc>();}}
};



static std::map<std::string, Activation> stringToActivationMap = {
    {"linear", Activation::Linear},
    {"sigmoid", Activation::Sigmoid},
    {"tanh", Activation::Tanh},
    {"softplus", Activation::Softplus},
    {"relu", Activation::Relu},
    {"prelu", Activation::Prelu},
    {"elu", Activation::Elu},
    {"pelu", Activation::Pelu},
    {"srelu", Activation::Srelu},
    {"gauss", Activation::Gauss},
    {"pgauss", Activation::Pgauss},
    {"sin", Activation::Sin},
    {"psin", Activation::Psin},
    {"sinc", Activation::Sinc},
    {"psinc", Activation::Psinc}
};



static std::map<Activation, std::string> activationToStringMap = {
    {Activation::Linear, "linear"},
    {Activation::Sigmoid, "sigmoid"},
    {Activation::Tanh, "tanh"},
    {Activation::Softplus, "softplus"},
    {Activation::Relu, "relu"},
    {Activation::Prelu, "prelu"},
    {Activation::Elu, "elu"},
    {Activation::Pelu, "pelu"},
    {Activation::Srelu, "srelu"},
    {Activation::Gauss, "gauss"},
    {Activation::Pgauss, "pgauss"},
    {Activation::Sin, "sin"},
    {Activation::Psin, "psin"},
    {Activation::Sinc, "sinc"},
    {Activation::Psinc, "psinc"}
};



} //namespace omnilearn



#endif //OMNILEARN_ACTIVATION_HH_