//burnet.hh

#ifndef BURNET_HH_
#define BURNET_HH_

#include <exception>
#include <string>
#include <vector>
#include <cmath>
#include <memory>



namespace burnet
{



//=============================================================================
//=============================================================================
//=============================================================================
//=== EXCEPTIONS DEFINITION ===================================================
//=============================================================================
//=============================================================================
//=============================================================================



struct Exception : public std::exception
{
    Exception(std::string const& msg):
    _msg("[EasyLearn.exception : " + msg + "]")
    {
    }

    virtual ~Exception()
    {
    }

    virtual const char* what() const noexcept
    {
        return _msg.c_str();
    }

private:
    std::string const _msg;
};



struct VectorSizeException : public Exception
{
    VectorSizeException(std::string const& msg):
    Exception(msg)
    {
    }

    virtual ~VectorSizeException()
    {
    }
};



struct AggregationException : public Exception
{
    AggregationException(std::string const& msg):
    Exception(msg)
    {
    }

    virtual ~AggregationException()
    {
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== UTILITY FUNCTIONS =======================================================
//=============================================================================
//=============================================================================
//=============================================================================



double dot(std::vector<double> const& a, std::vector<double> const& b)
{
    if(a.size() != b.size())
    {
        throw VectorSizeException("In a dot product, both vectors must have the same number of element.");
    }

    double result = 0;
    for(unsigned i = 0; i < a.size(); i++)
    {
        result += (a[i] * b[i]);
    }
    return result;
}


double distance(std::vector<double> const& a, std::vector<double> const& b, double order = 2)
{
    if(a.size() != b.size())
    {
        throw VectorSizeException("To calculate the dispance between two vectors, they must have the same number of element.");
    }

    double result = 0;
    for(unsigned i=0; i<a.size(); i++)
    {
        result += std::pow((a[i] - b[i]), order);
    }
    return std::pow(result, 1/order);
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== AGGREGATION =============================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Aggregation //abstract class
{
    friend class Neuron;
public:
    Aggregation(unsigned k):
    _k(k),
    _weights(std::vector<std::vector<double>>()),
    _bias(std::vector<double>())
    {
        if (_k == 0)
        {
            throw AggregationException("An aggregation structure must manage at least one wheight set.");
        }
    }

    Aggregation(std::vector<std::vector<double>> const& weights, std::vector<double> const& bias):
    _k(_weights.size()),
    _weights(weights),
    _bias(bias)
    {
        if (_k == 0)
        {
            throw AggregationException("An aggregation structure must handle at least one weight set.");
        }
        if (_weights.size() != _bias.size())
        {
            throw AggregationException("An aggregation structure must handle the same amount of weight set and bias.");
        }
    }

    virtual ~Aggregation();
    virtual std::pair<double, unsigned> aggregate(std::vector<double> const& inputs) = 0; //double is the result, unsigned is the index of the weight set used
    virtual std::vector<double> prime(std::vector<double> const& inputs, unsigned index) = 0; //return derivatives according to each weight (weights from the index "index")
    virtual void learn(double gradient, double learningRate, double momentum) = 0;

    double k() const
    {
        return _k;
    }

    std::vector<std::vector<double>> weights() const
    {
        return _weights;
    }

    std::vector<double> bias() const
    {
        return _bias;
    }

private:
    virtual std::pair<std::vector<double>&, double&> weightRef(unsigned index) final
    {
        return {_weights[index], _bias[index]};
    }

protected:
    unsigned const _k; // number of weight set
    std::vector<std::vector<double>> _weights;
    std::vector<double> _bias;
};



class Dot : public Aggregation
{
public:
    Dot(): Aggregation(1)
    {
    }

    Dot(std::vector<std::vector<double>> weights, std::vector<double> bias):
    Aggregation(weights, bias)
    {
        if (_k > 1)
        {
            throw AggregationException("The burnet::Dot aggregation structure must handle exactly one weight set.");
        }
    }

    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs)
    {
        double result = 0;
        for(unsigned i = 0; i < _weights[0].size(); i++)
        {
            result += (_weights[0][i] * inputs[i]);
        }
        return {dot(inputs, _weights[0]), 0};
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] unsigned index)
    {
        return inputs;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



class Distance : public Aggregation
{
public:
    Distance(unsigned order = 2):
    Aggregation(1),
    _order(order)
    {
    }


    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs)
    {
        return {distance(inputs, _weights[0], _order), 0};
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] unsigned index)
    {
        double a = std::pow(aggregate(inputs).first, (1-_order));
        std::vector<double> result(_weights[0].size(), 0);

        for(unsigned i = 0; i < _weights[0].size(); i++)
        {
            result[i] += (-std::pow((inputs[i] - _weights[0][i]), _order-1) * a);
        }
        return result;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }

protected:
    unsigned const _order;
};



class Maxout : public Aggregation
{
public:
    Maxout(unsigned k):
    Aggregation(k)
    {
    }


    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs)
    {
        std::vector<double> dots(_k, 0);

        for(unsigned i = 0; i < _k; i++)
        {
            dots[i] = dot(inputs, _weights[i]);
        }

        std::pair<double, unsigned> max = {dots[0], 0};
        for(unsigned i = 0; i < _k; i++)
        {
            if(dots[i] > max.first)
            {
                max.first = dots[i];
                max.second = i;
            }
        }

        return max;
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] unsigned index)
    {
        return inputs;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== ACTIVATION ==============================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Activation //interface
{
public:
    virtual ~Activation();
    virtual double activate(double val) = 0;
    virtual double prime(double val) = 0;
    virtual void learn(double gradient, double learningRate, double momentum) = 0;
};



class Linear : public Activation
{
    double activate(double val)
    {
        return val;
    }

    double prime([[maybe_unused]] double val)
    {
        return 1;
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



class Sigmoid : public Activation
{
    double activate(double val)
    {
        return 1 / (1 + std::exp(-val));
    }

    double prime(double val)
    {
        return val * (1 - val);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



class  Tanh : public Activation
{
    double activate(double val)
    {
        return std::tanh(val);
    }

    double prime(double val)
    {
        return -1/std::pow(std::cosh(val),2);
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



class Softplus : public Activation
{
    double activate(double val)
    {
        return std::log(std::exp(val) + 1);
    }

    double prime(double val)
    {
        return 1 / (1 + std::exp(-val));
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



class Relu : public Activation //if coef != 0, it is equivalent to leakyRelu
{
    Relu(double coef = 0) : _coef(coef)
    {
    }

    double activate(double val)
    {
        return (val < 0 ? _coef*val : val);
    }

    double prime(double val)
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



//like Relu, but the coef is learnable
class Prelu : public Activation
{
    Prelu(double coef = 0) : _coef(coef)
    {
    }

    double activate(double val)
    {
        return (val < 0 ? _coef*val : val);
    }

    double prime(double val)
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



class Elu : public Activation
{
    Elu(double coef = 0) : _coef(coef)
    {
    }

    double activate(double val)
    {
        return (val < 0 ? _coef*(std::exp(val)-1) : val);
    }

    double prime(double val)
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



//like Elu, but the coef is learnable
class Pelu : public Activation
{
    Pelu(double coef = 0) : _coef(coef)
    {
    }

    double activate(double val)
    {
        return (val < 0 ? _coef*(std::exp(val)-1) : val);
    }

    double prime(double val)
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



// S-shaped rectified linear unit (combination of 3 linear functions, with 4 learnable parameters )
class Srelu : public Activation
{
    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }

protected:
    double _coef1;
    double _coef2;
    double _coef3;
};



class Gauss : public Activation
{
    double activate(double val)
    {
        return std::exp(-std::pow(val, 2));
    }

    double prime(double val)
    {
        return -2 * val * std::exp(-std::pow(val, 2));
    }

    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



class Softexp : public Activation
{
    double activate(double val)
    {
        if(_coef < 0)
            return -std::log(1-(_coef*(val + _coef))) / _coef;
        if(_coef == 0)
            return val;
        if(_coef > 0)
            return ((std::exp(_coef * val) - 1) / _coef) + _coef;
    }

    double prime(double val)
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



//=============================================================================
//=============================================================================
//=============================================================================
//=== NEURON ==================================================================
//=============================================================================
//=============================================================================
//=============================================================================


class Neuron
{
protected:
    std::shared_ptr<Activation> _Activation;   //should be a value but polymorphism is needed
    std::shared_ptr<Aggregation> _Aggregation; //should be a value but polymorphism is needed
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== LAYER ===================================================================
//=============================================================================
//=============================================================================
//=============================================================================







//=============================================================================
//=============================================================================
//=============================================================================
//=== PERCEPTRON ==============================================================
//=============================================================================
//=============================================================================
//=============================================================================




} // namespace burnet


#endif //BURNET_HH_