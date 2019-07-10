#ifndef BURNET_NEURON_HH_
#define BURNET_NEURON_HH_

#include "Activation.hh"
#include "Aggregation.hh"

#include <memory>
#include <type_traits>

namespace burnet
{



template<typename Act_t, typename Aggr_t,
typename = typename std::enable_if<
std::is_base_of<Activation, Act_t>::value &&
std::is_base_of<Aggregation, Aggr_t>::value,
void>::type>
class Neuron
{
public:
    Neuron(Act_t const& activation, Aggr_t const& aggregation):
    _activation(activation),
    _aggregation(aggregation),
    _initialized(false),
    _weights(),
    _bias(),
    _inputs(),
    _aggregResults(),
    _actResults()
    {
    }


    Neuron(Act_t const& activation, Aggr_t const& aggregation, Matrix weights, std::vector<double> const& bias):
    _activation(activation),
    _aggregation(aggregation),
    _initialized(true),
    _weights(weights),
    _bias(bias),
    _inputs(),
    _aggregResults(),
    _actResults()
    {
    }


    void init(unsigned batchSize)
    {
        _aggregResults = std::vector<std::pair<double, unsigned>>(batchSize, {0.0, 0});
        _actResults = std::vector<double>(batchSize, 0);
    }


    void initWeights()
    {
        if(_initialized)
            return;
    }


    //each line of the input matrix is a feature of the batch. Returns one result per feature.
    std::vector<double> process(Matrix const& inputs)
    {
        std::vector<double> results(inputs.size(), 0);

        for(unsigned i = 0; i < inputs.size(); i++)
        {
            results[i] = _activation.activate(_aggregation.aggregate(inputs[i], _weights, _bias).first);
        }
    }


    //each line of the input matrix is a feature of the batch. Returns one result per feature.
    std::vector<double> processToLearn(Matrix const& inputs)
    {
        _inputs = inputs;

        //dropConnect
        if(drop > std::numeric_limits<double>::epsilon())
        {
            double denominator = 1 - drop;
            for(unsigned i=0; i<_inputs.size(); i++)
            {
                for(unsigned j=0; j<_inputs[0].size(); i++)
                {
                    if(dropDist(dropGen))
                        _inputs[i][j] = 0;
                    else
                        _inputs[i][j] /= denominator;
                }
            }
        }

        //processing
        for(unsigned i = 0; i < inputs.size(); i++)
        {
            _aggregResults[i] = _aggregation.aggregate(inputs[i], _weights, _bias);
            _actResults[i] = _activation.activate(_aggregResults[i].first);
        }

        return _actResults;
    }


    //one input gradient per feature
    void computeGradient(std::vector<double> inputGradients)
    {

    }


    void updateWeights(double learningRate, double L1, double L2, double tackOn, double maxNorm, double momentum)
    {

    }


    static initDropConnect(double dropConnect, unsigned seed)
    {
        drop = dropConnect;

        if(seed == 0)
            seed = static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count());
        dropGen = std::mt19937(seed);

        dropDist = std::bernoulli_distribution(dropConnect);
    }

protected:
    static double drop;
    static std::mt19937 dropGen;
    static std::bernoulli_distribution dropDist;

    Act_t _activation;
    Aggr_t _aggregation;

    bool _initialized;
    Matrix _weights;
    std::vector<double> _bias;

    Matrix _inputs;
    std::vector<std::pair<double, unsigned>> _aggregResults;
    std::vector<double> _actResults;

};



} //namespace burnet



#endif //BURNET_NEURON_HH_