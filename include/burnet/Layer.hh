#ifndef BURNET_LAYER_HH_
#define BURNET_LAYER_HH_

#include "Neuron.hh"

namespace burnet
{


class ILayer
{
public:
    virtual ~ILayer();
    virtual Matrix process(Matrix const& inputs) const = 0;
    virtual Matrix processToLearn(Matrix const& inputs) = 0;
    virtual std::vector<double> getGradients() = 0;
    virtual unsigned size() const = 0;
    virtual void init(unsigned nbInputs, unsigned nbOutputs, unsigned batchSize) = 0;


    static void initDropout(unsigned seed, double drop)
    {
        dropout = drop;

        if(seed == 0)
            seed = static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count());
        dropGen = std::mt19937(seed);
        dropoutSeed = seed;

        dropDist = std::bernoulli_distribution(drop);
    }


protected:
    static double dropoutSeed;
    static double dropout;
    static std::mt19937 dropGen;
    static std::bernoulli_distribution dropDist;
};


//=============================================================================
//=============================================================================
//=============================================================================
//=== LAYER ===================================================================
//=============================================================================
//=============================================================================
//=============================================================================


template<typename Act_t, typename Aggr_t,
typename = typename std::enable_if<
std::is_base_of<Activation, Act_t>::value &&
std::is_base_of<Aggregation, Aggr_t>::value,
void>::type>
class Layer : public ILayer
{
public:
    Layer(LayerParam const& param, std::vector<Neuron<Act_t, Aggr_t>> neurons = std::vector<Neuron<Act_t, Aggr_t>>()):
    _distrib(param.distrib),
    _distVal1(param.distribVal1),
    _distVal1(param.distribVal2),
    _maxNorm(param.maxNorm),
    _k(param.k),
    _neurons(neurons.size() == 0 ? std::vector<Neuron<Act_t, Aggr_t>>(param.size) : neurons)
    {
    }


    Matrix process(Matrix const& inputs) const
    {
        Matrix output(inputs.size(), {inputs[0].size(), 0});
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            std::vector<double> result = _neurons.process(inputs);
            for(unsigned j = 0; j < result.size(); j++)
            {
                output[j][i] = result[j];
            }
        }
        return output;
    }


    Matrix processToLearn(Matrix const& inputs)
    {
        Matrix output(inputs.size(), {inputs[0].size(), 0});
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            std::vector<double> result = _neurons.processToLearn(inputs);
            for(unsigned j = 0; j < result.size(); j++)
            {
                output[j][i] = result[j];
                //dropOut
                if(dropout > std::numeric_limits<double>::epsilon())
                {
                    if(dropDist(dropGen))
                        output[j][i] = 0;
                    else
                        output[j][i] /= (1-dropout);
                }
            }
        }
        return output;
    }


    //one gradient per input
    std::vector<double> getGradients()
    {
        std::vector<double> grad(_inputSize, 0);

        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            std::vector<double> neuronGrad = _neurons[i].getGradients();
            for(unsigned j = 0; j < _inputSize; j++)
                grad[j] += neuronGrad[j];
        }

        return grad;
    }


    unsigned size() const
    {
        return _neurons.size();
    }


    void init(unsigned nbInputs, unsigned nbOutputs, unsigned batchSize)
    {
        _inputSize = nbInputs;
        for(unsigned i = 0; i < size(); i++)
        {
            _neurons[i].init(_distrib, _distVal1, _distVal2, nbInputs, nbOutputs, batchSize, _k);
        }
    }

protected:

    unsigned const _inputSize;
    Distrib _distrib;
    double _distVal1; //mean (if uniform), boundary (if uniform)
    double _distVal2; //deviation (if normal) or useless (if uniform)
    double const _maxNorm;
    unsigned _k; //number of weight set for each neuron
    std::vector<Neuron<Act_t, Aggr_t>> _neurons;
};



} //namespace burnet



#endif //BURNET_LAYER_HH_