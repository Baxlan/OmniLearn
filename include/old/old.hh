//old.hH

#ifndef OLD_HH_
#define OLD_HH_

#include "NNfunctions.hh"

#include <vector>
#include <random>
#include <chrono>
#include <utility>
#include <exception>
#include <string>



namespace stb
{





namespace learn
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

    std::string const _msg;
};


struct ProbabilityException : public Exception
{
    ProbabilityException(std::string const& msg):
    Exception(msg)
    {
    }


    virtual ~ProbabilityException()
    {
    }
};


struct InputException : public Exception
{
    InputException(std::string const& msg):
    Exception(msg)
    {
    }


    virtual ~InputException()
    {
    }
};


struct ValueException : public Exception
{
    ValueException(std::string const& msg):
    Exception(msg)
    {
    }


    virtual ~ValueException()
    {
    }
};


//=============================================================================
//=============================================================================
//=============================================================================
//=== INITIALIZER STRUCTURES ==================================================
//=============================================================================
//=============================================================================
//=============================================================================



struct NeuralFunctions
{
    NeuralFunctions(
    double (* _activation)(double),
    double (* _activationPrime)(double),
    double (* _aggregation)(std::vector<double> const&, std::vector<double> const&) = stb::learn::dot,
    double (* _aggregationPrime)(std::vector<double> const&, std::vector<double> const&, unsigned) = stb::learn::dotPrime):
    activation(_activation),
    activationPrime(_activationPrime),
    aggregation(_aggregation),
    aggregationPrime(_aggregationPrime)
    {
    }

    double (* activation)(double);
    double (* activationPrime)(double);
    double (* aggregation)(std::vector<double> const&, std::vector<double> const&);
    double (* aggregationPrime)(std::vector<double> const&, std::vector<double> const&, unsigned);
};



struct WeightInitializer
{
    WeightInitializer(double _mean=0, double _deviation=1, int _seed=0, double _bias=0):
    mean(_mean),
    deviation(_deviation),
    seed(_seed),
    bias(_bias)
    {
        if(deviation < 0)
            throw ValueException("A deviation shouldn't be negative.");
    }

    double mean;
    double deviation;
    int seed;
    double bias;
};



struct NeuronProportion
{
    NeuronProportion(
    NeuralFunctions const& _func,
    WeightInitializer const& _init = WeightInitializer(),
    double _proportion = 1):
    func(_func),
    init(_init),
    proportion(_proportion)
    {
        if(proportion < 0)
            throw ValueException("A  proportion shouldn't be negative.");
    }

    NeuralFunctions func;
    WeightInitializer init;
    double proportion;
};



struct LayerSettings
{
    LayerSettings(double _dropOut = 0, double _dropConnect = 0, double _maxNorm = 0):
    dropOut(_dropOut),
    dropConnect(_dropConnect),
    maxNorm(_maxNorm)
    {
        if(dropOut < 0 || dropOut >= 1)
            throw ProbabilityException("DropOut must be between 0 included and 1 excluded.");
        if(dropConnect < 0 || dropConnect >= 1)
            throw ProbabilityException("DropConnect must be between 0 included and 1 excluded.");
    }

    double dropOut;
    double dropConnect;
    double maxNorm;
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

public:

    Neuron(NeuralFunctions const& functions, std::vector<double> weights, double bias):
    activate(functions.activation),
    activatePrime(functions.activationPrime),
    aggregate(functions.aggregation),
    aggregatePrime(functions.aggregationPrime),
    _weights(weights),
    _bias(bias),
    _inputs(std::vector<double>()),
    _result(0)
    {
    }


    Neuron(NeuralFunctions const& functions, WeightInitializer const& initializer, unsigned nbInputs):
    activate(functions.activation),
    activatePrime(functions.activationPrime),
    aggregate(functions.aggregation),
    aggregatePrime(functions.aggregationPrime),
    _weights(initWeights(nbInputs, initializer.mean, initializer.deviation, initializer.seed)),
    _bias(initializer.bias),
    _inputs(std::vector<double>()),
    _result(0)
    {
    }


    static std::vector<double> initWeights(unsigned nbInputs, double mean, double deviation, int seed)
    {
        if(deviation < 0)
            throw ValueException("A deviation shouldn't be negative.");

        //if seed == 0, generate a "random" seed
        if(seed == 0)
            seed = static_cast<int>(std::chrono::steady_clock().now().time_since_epoch().count());
        std::mt19937 generator(seed); //uniform law
        std::normal_distribution<double> distribution(mean, deviation); //normal law

        std::vector<double> w(nbInputs);
        for(unsigned i=0 ; i<nbInputs ; i++)
            w[i] = distribution(generator);
        return w;
    }


    double process(std::vector<double> const& values)
    {
        if(values.size() != _weights.size())
            throw InputException("The number of inputs is different of the number of weights.");
        _inputs=values;

        double sum = aggregate(values, _weights) + _bias;
        _result = activate(sum);
        return _result;
    }


    double processToLearn(std::vector<double> const& values, double dropConnect = 0)
    {
        if(values.size() != _weights.size())
            throw InputException("The number of inputs is different of the number of weights.");
        _inputs=values;

        //dropConnect
        if(dropConnect > std::numeric_limits<double>::epsilon())
        {
            std::mt19937 generator(static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count())); //uniform law
            std::bernoulli_distribution distribution(dropConnect); //uniform law
            for(unsigned i=0; i<_inputs.size(); i++)
            {
                if(distribution(generator))
                    _inputs[i] = 0;
            }
        }

        //process
        double sum = aggregate(values, _weights) + _bias;
        _result = activate(sum);
        return _result;
    }


    void updateWeights(double delta, double learningRate, double L1Regularization, double L2Regularization, double tackOn, double maxNorm, double momentum = 0)
    {
        for(unsigned i=0; i<_weights.size();i++)
        {
            _weights[i] += (learningRate*(_inputs[i] * delta * activatePrime(_result)
            + L2Regularization*_weights[i] + L1Regularization) + tackOn + momentum * 0);

            _bias += learningRate * delta;
        }
        //max norm constraint
        double norm = std::sqrt(quadraticSum(_weights));
        if(maxNorm > 0 && norm > maxNorm)
        {
            for(unsigned i=0; i<_weights.size(); i++)
            {
                _weights[i] *= (maxNorm/norm);
            }
        }
    }


    std::vector<double> getWeights() const
    {
        return _weights;
    }


    double getBias() const
    {
        return _bias;
    }


    double getAbsoluteWeightsSum() const
    {
        return absoluteSum(_weights);
    }


    double getSquaredWeightsSum() const
    {
        return quadraticSum(_weights);
    }


protected:
    //Activation function
    double (* activate)(double);
    //Derivative of the activation function
    double (* activatePrime)(double);
    //agreggation function
    double (* aggregate)(std::vector<double> const&, std::vector<double> const&);
    //derivative of the aggregation function (according to the i-th weight)
    double (* aggregatePrime)(std::vector<double> const&, std::vector<double> const&, unsigned);
    //weight of each input
    std::vector<double> _weights;
    //bias parameter
    double _bias;
    //last inputs
    std::vector<double> _inputs;
    //last result emited
    double _result;
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
    Layer(std::vector<Neuron> const& neurons, LayerSettings const& settings = LayerSettings()):
    _neurons(neurons),
    _delta(std::vector<double>(0)),
    _dropOut(settings.dropOut),
    _dropConnect(settings.dropConnect),
    _maxNorm(settings.maxNorm)
    {
    }


    Layer(std::vector<NeuronProportion> const& prop, unsigned nbNeurons, unsigned nbInputs, LayerSettings const& settings = LayerSettings()):
    _neurons(initNeurons(prop, nbNeurons, nbInputs)),
    _delta(std::vector<double>(_neurons.size(), 0)),
    _dropOut(settings.dropOut),
    _dropConnect(settings.dropConnect),
    _maxNorm(settings.maxNorm)
    {
    }


    static std::vector<Neuron> initNeurons(std::vector<NeuronProportion> prop, unsigned nbNeurons, unsigned nbInputs)
    {
        //normalize proportions to 1
        double sum = 0;
        for(NeuronProportion const& p : prop)
        {
            if(p.proportion < 0)
                throw ValueException("A proportion shouldn't be negative.");
            sum += p.proportion;
        }
        if(sum < 1 || sum > 1)
        {
            for(unsigned i = 0; i<prop.size(); i++)
                prop[i].proportion /= sum;
        }

        std::vector<unsigned> nbOfEachNeuron(prop.size(), 0);
        for(unsigned i = 0; i<prop.size(); i++)
            nbOfEachNeuron[i] = static_cast<unsigned>(prop[i].proportion * static_cast<double>(nbNeurons));

        std::vector<Neuron> neurons;
        for(unsigned i=0 ; i<nbNeurons ; i++)
        {
            unsigned propIndex = 0;
            unsigned sum2 = 0;
            for(; propIndex < nbOfEachNeuron.size(); propIndex++)
            {
                sum2 += nbOfEachNeuron[propIndex];
                if(i <= sum2)
                    break;
            }
            neurons.push_back(Neuron(prop[propIndex].func, prop[propIndex].init, nbInputs));
        }
        return neurons;
    }


    std::vector<double> process(std::vector<double> inputs)
    {
        std::vector<double> outputs(_neurons.size(), 0);
        for(unsigned i=0; i<_neurons.size(); i++)
        {
            outputs[i] = _neurons[i].process(inputs);
        }
        return outputs;
    }


    std::vector<double> processToLearn(std::vector<double> inputs)
    {
        std::vector<double> outputs(_neurons.size(), 0);

        std::mt19937 generator(static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count())); //uniform law
        std::bernoulli_distribution distribution(_dropOut); //uniform law

        double denominator = 1 - _dropOut;
        for(unsigned i=0; i<_neurons.size(); i++)
        {
            outputs[i] = _neurons[i].processToLearn(inputs, _dropConnect);
            //dropOut
            if(std::abs(_dropOut) > std::numeric_limits<double>::epsilon())
            {
                outputs[i] /= denominator;
                if(distribution(generator))
                    outputs[i] = 0;
            }
        }
        return outputs;
    }


    // returns the previous neurons's partial delta (to multiply by their calculus)
    // ==> backpropagation
    std::vector<double> getNextDelta() const
    {
        std::vector<double> delta(_neurons[0].getWeights().size(), 0);

        for(unsigned weight=0; weight<delta.size(); weight++)
        {
            for(unsigned i=0; i<_neurons.size(); i++)
                delta[weight] += (_delta[i] * _neurons[i].getWeights()[weight]);
        }
        return delta;
    }


    void updateWeights(double learningRate, double L1Regularization, double L2Regularization, double tackOn, double maxNorm)
    {
        for(unsigned i=0; i<_neurons.size(); i++)
            _neurons[i].updateWeights(_delta[i], learningRate, L1Regularization, L2Regularization, tackOn, maxNorm);
    }


    std::vector<Neuron> getNeurons() const
    {
        return _neurons;
    }


    void setDelta(std::vector<double> const& delta)
    {
        if(delta.size() != _delta.size())
            throw InputException("The number of deltas is different of the number of neurons.");
        _delta = delta;
    }


    double getAbsoluteWeightsSum()
    {
        double sum = 0;
        for(unsigned i=0; i<_neurons.size(); i++)
            sum += _neurons[i].getAbsoluteWeightsSum();
        return sum;
    }


    double getSquaredWeightsSum()
    {
        double sum = 0;
        for(unsigned i=0; i<_neurons.size(); i++)
            sum += _neurons[i].getSquaredWeightsSum();
        return sum;
    }


protected:
    // neurons of the layer
    std::vector<Neuron> _neurons;
    // store the delta of each neuron
    std::vector<double> _delta;
    // probability to shutdown a neuron
    double _dropOut;
    // probability to temporary set a weight to 0
    double _dropConnect;
    // the maximum norm a weight vector can have. if 0, maxNorm is not applied
    double _maxNorm;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== NETWORK =================================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Network
{
public:

protected:
    std::vector<std::vector<Layer>> _layers;
    double _batchSize;
    double _learningRate;
    // L1 regularization strenght
    double _L1reg;
    // L1 regularization, not multiplied by the learning rate
    double _tackOn;
    // L2 regularization strenght
    double _L2reg;
    // if loss < margin, then loss = 0
    double _margin;
};



} // namespace learn



} // namespace stb



#endif //OLD_HH_