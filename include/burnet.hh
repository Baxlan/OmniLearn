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


struct ProcessException : public Exception
{
    ProcessException(std::string const& msg):
    Exception(msg)
    {
    }

    virtual ~ProcessException()
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


double average(std::vector<double> const& a)
{
    double sum = 0;
    for(double const& val : a)
    {
        sum += val;
    }
    return sum / a.size();
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
        return {dot(inputs, _weights[0]) + _bias[0], 0};
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
            result[i] += (-std::pow((inputs[i] - _weights[0][i]) + _bias[0], _order-1) * a);
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
            dots[i] = dot(inputs, _weights[i]) + _bias[i];
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



// if there are two hinges, then this is S-shaped rectified linear unit (Srelu)
class Srelu : public Activation
{
    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }

protected:

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
    double process(std::vector<double> const& inputs)
    {
        return _Activation->activate(_Aggregation->aggregate(inputs).first);
    }


    double processToLearn(std::vector<double> const& inputs)
    {
        if(_currentFeature >= (_batchSize - 1))
        {
            throw ProcessException("Batch size have been reashed, gradients must be calculated then parameters must be updated before processing new features.");
        }
        _currentFeature++;
        _aggregResults[_currentFeature-1] = _Aggregation->aggregate(inputs);
        _activResults[_currentFeature-1] = _Activation->activate(_aggregResults[_currentFeature-1].first);
        return _activResults[_currentFeature-1];
    }


    void processGradient(std::vector<double> inputGradients) //one input by feature in the batch
    {
        if(_currentFeature < (_batchSize - 1))
        {
            throw ProcessException("Calculating gradient but the batch size have not been reashed.");
        }
        else if(_currentFeature > (_batchSize - 1))
        {
            throw ProcessException("Calculating gradient but parameters have not be updated.");
        }

        std::vector<double> actGradients(_batchSize, 0); //store gradients between activation and aggregation for each feature of the batch

        for(_currentFeature = 0; _currentFeature < _batchSize; _currentFeature++)
        {
            _inputGradients[_currentFeature] = inputGradients[_currentFeature];
            actGradients[_currentFeature] = _inputGradients[_currentFeature] * _Activation->prime(_activResults[_currentFeature]); // * aggreg.prime ?
            _averageActGradient = average(actGradients);
        }

        //setting all partial gradients on 0
        _gradients = std::vector<std::vector<double>>(_Aggregation->k(), std::vector<double>(_batchSize, 0));

        //storing new partial gradients
        for(_currentFeature = 0; _currentFeature < _currentFeature; _currentFeature++)
        {
            _gradients[_aggregResults[_currentFeature].second][_currentFeature] = _aggregResults[_currentFeature].first * actGradients[_currentFeature];
        }

        _currentFeature++;
    }


    void updateWeights(double learningRate, double L1, double L2, double tackOn, double maxNorm, double momentum)
    {
        if(_currentFeature <= _batchSize - 1)
        {
            throw ProcessException("Updating parameters but gradients have not been calculated.");
        }

        _Activation->learn(average(_inputGradients), 0, 0);
        _Aggregation->learn(_averageActGradient, 0, 0);

        for(unsigned i = 0; i < _Aggregation->k(); i++)
        {
            std::pair<std::vector<double>&, double&> w = _Aggregation->weightRef(i);
            double gradient = average(_gradients[i]);

            for(unsigned j = 0; j < w.first.size(); j++)
            {
                w.first[j] += (learningRate*(gradient + (L2*w.first[j]) + L1) + tackOn);
            }
            w.second += learningRate * gradient; // to divide by inputs
        }
    }

protected:
    std::shared_ptr<Activation> _Activation;   //should be a value but polymorphism is needed
    std::shared_ptr<Aggregation> _Aggregation; //should be a value but polymorphism is needed

    unsigned const _batchSize;

    unsigned _currentFeature; // current feature number in mini-batch
    std::vector<std::pair<double, unsigned>> _aggregResults; // results obtained by aggregation and weight set used, for each feature in the batch
    std::vector<double> _activResults; // results obtained by activation, for each feature in the batch

    std::vector<std::vector<double>> _previousWeightUpdate; // previous update aplied to each weight in each weight set
    std::vector<double> _previousBiasUpdate; // previous update aplied to bias for each weight set
    std::vector<double> _inputGradients; //for each feature of the batch, gradients entered from previous layers
    double _averageActGradient; //averaged gradient over all features of the batch, between activation and aggregation
    std::vector<std::vector<double>> _gradients; // partial gradient obtained for each feature of the batch and for each weight set
    //weight set //feature //partial gradient
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== LAYER ===================================================================
//=============================================================================
//=============================================================================
//=============================================================================


class Layer
{
public:
    std::vector<double> process()
    {

    }

    std::vector<double> processToLearn()
    {

    }

    void processGradients()
    {

    }

    void updateWeights()
    {

    }

protected:
    double const _dropout;
    double const _dropconnect;
    double const _maxNorm;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== NETWORK ==================================================================
//=============================================================================
//=============================================================================
//=============================================================================


class Network
{
public:


protected:
    double _learningRate;
    double const _L1;
    double const _L2;
    double const _tackOn;
    double const _maxEpoch;
    double _currentEpoch;
};


//=============================================================================
//=============================================================================
//=============================================================================
//=== PERCEPTRON ==============================================================
//=============================================================================
//=============================================================================
//=============================================================================




} // namespace burnet


#endif //BURNET_HH_