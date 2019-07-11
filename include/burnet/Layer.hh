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


protected:

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
            }
        }
        return output;
    }

    double const _maxNorm;
    std::vector<Neuron<Act_t, Aggr_t>> _neurons;
};



} //namespace burnet



#endif //BURNET_LAYER_HH_