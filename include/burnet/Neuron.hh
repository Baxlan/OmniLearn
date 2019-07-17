#ifndef BURNET_NEURON_HH_
#define BURNET_NEURON_HH_

#include "Activation.hh"
#include "Aggregation.hh"

#include <memory>
#include <type_traits>

namespace burnet
{


//=============================================================================
//=============================================================================
//=============================================================================
//=== NEURON ==================================================================
//=============================================================================
//=============================================================================
//=============================================================================


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
    _bias(0),
    _inputs(),
    _aggregResults(),
    _actResults(),
    _inputGradients(),
    _actGradients(),
    _gradients()
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
    _actResults(),
    _inputGradients(),
    _actGradients(),
    _gradients(Matrix(weights.size(), std::vector<double>(weights[0].size(), 0)))
    {
    }


    void init(unsigned batchSize)
    {
        _aggregResults = std::vector<std::pair<double, unsigned>>(batchSize, {0.0, 0});
        _actResults = std::vector<double>(batchSize, 0);
        _actGradients = std::vector<double>(batchSize, 0);
    }


    void initWeights()
    {
        if(_initialized)
            return;
        //init _weights, _inputGradients and _gradients (size)
    }


    //each line of the input matrix is a feature of the batch. Returns one result per feature.
    std::vector<double> process(Matrix const& inputs) const
    {
        std::vector<double> results(inputs.size(), 0);

        for(unsigned i = 0; i < inputs.size(); i++)
        {
            results[i] = _activation.activate(_aggregation.aggregate(inputs[i], _weights, _bias).first);
        }
        return results;
    }


    //each line of the input matrix is a feature of the batch. Returns one result per feature.
    std::vector<double> processToLearn(Matrix const& inputs)
    {
        _inputs = inputs;

        //dropConnect
        if(dropConnect > std::numeric_limits<double>::epsilon())
        {
            for(unsigned i=0; i<_inputs.size(); i++)
            {
                for(unsigned j=0; j<_inputs[0].size(); i++)
                {
                    if(dropDist(dropGen))
                        _inputs[i][j] = 0;
                    else
                        _inputs[i][j] /= (1 - dropConnect);
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
        _inputGradients = inputGradients;
        std::vector<unsigned> setCount(_weights.size(), 0); //store the amount of feature that passed through each weight set

        for(unsigned feature = 0; feature < _actResults.size(); feature++)
        {
            _actGradients[feature] = _activation.prime(_actResults[feature]) * _inputGradients[feature];
            std::vector<double> grad(_aggregation.prime(_inputs[feature], _weights[_aggregResults[feature].second]));

            for(unsigned i = 0; i < grad.size(); i++)
            {
                _gradients[_aggregResults[feature].second][i] += (_actGradients[feature]*grad[i]);
            }
            setCount[_aggregResults[feature].second]++;
        }

        //average gradients over features
        for(unsigned i = 0; i < _gradients.size(); i++)
        {
            for(unsigned j = 0; j < _gradients[0].size(); j++)
            {
                _gradients[i][j] /= setCount[i];
            }
        }
    }


    void updateWeights(double learningRate, double L1, double L2, double tackOn, double maxNorm, double momentum)
    {
        double averageInputGrad = 0;
        for(unsigned i = 0; i < _inputGradients.size(); i++)
        {
            averageInputGrad += _inputGradients[i];
        }
        averageInputGrad /= _inputGradients.size();

        double averageActGrad = 0;
        for(unsigned i = 0; i < _actGradients.size(); i++)
        {
            averageActGrad += _actGradients[i];
        }
        averageActGrad /= _actGradients.size();

        _activation.learn(averageInputGrad, learningRate, momentum);
        _aggregation.learn(averageActGrad, learningRate, momentum);

        for(unsigned i = 0; i < _weights.size(); i++)
        {
            for(unsigned j = 0; j < _weights[0].size(); j++)
            {
                _weights[i][j] += (learningRate*(_gradients[i][j] + (L2 * _weights[i][j]) + L1) + tackOn);
            }
        }
    }


    //one gradient per input
    std::vector<double> getGradients() const
    {
        std::vector<double> grad(_weights[0].size(), 0);

        for(unsigned i = 0; i < _weights.size(); i++)
        {
            for(unsigned j = 0; j < _weights[0].size(); j++)
            {
                grad[j] += _gradients[i][j] * _weights[i][j];
            }
        }
        return grad;
    }


    static void initDropConnect(double dropCo, unsigned seed)
    {
        dropConnect = dropCo;

        if(seed == 0)
            seed = static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count());
        dropGen = std::mt19937(seed);

        dropDist = std::bernoulli_distribution(dropCo);
    }


protected:
    static double dropConnect;
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

    std::vector<double> _inputGradients; //gradient from next layer for each feature af the batch
    std::vector<double> _actGradients; //gradient between aggregation and activation
    Matrix _gradients; //sum (over all features of the batch) of partial gradient for each weight

};



} //namespace burnet



#endif //BURNET_NEURON_HH_