#ifndef BURNET_LAYER_HH_
#define BURNET_LAYER_HH_

#include "Neuron.hh"

namespace burnet
{



class Layer
{
public:


protected:
    std::vector<Neuron> _neurons;

    double const _maxNorm;
};



} //namespace burnet



#endif //BURNET_LAYER_HH_