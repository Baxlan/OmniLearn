#ifndef BURNET_LAYER_HH_
#define BURNET_LAYER_HH_

#include "Neuron.hh"

namespace burnet
{



template<typename Act_t, typename Aggr_t,
typename = typename std::enable_if<
std::is_base_of<Activation, Act_t>::value &&
std::is_base_of<Aggregation, Aggr_t>::value,
void>::type>
class Layer
{
public:
    Matrix process(Matrix const& inputs)
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


    static void initDropout(double drop, unsigned seed)
    {
        dropout = drop;

        if(seed == 0)
            seed = static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count());
        dropGen = std::mt19937(seed);

        dropDist = std::bernoulli_distribution(drop);
    }


protected:
    static double dropout;
    static std::mt19937 dropGen;
    static std::bernoulli_distribution dropDist;

    unsigned const _inputSize;
    double const _maxNorm;
    std::vector<Neuron<Act_t, Aggr_t>> _neurons;
};



} //namespace burnet



#endif //BURNET_LAYER_HH_