#ifndef BURNET_NETWORK_HH_
#define BURNET_NETWORK_HH_

#include "Layer.hh"

namespace burnet
{

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
  Network(std::vector<std::pair<std::vector<double>, std::vector<double>>> data, NetworkParam const& param):
  _layers(),
  _batchSize(param.batchSize),
  _learningRate(param.learningRate),
  _maxEpoch(param.maxEpoch),
  _epochAfterOptimal(param.epochAfterOptimal),
  _trainData(data),
  _validationData(),
  _testData(),
  _epoch(0),
  _optimalEpoch(0)
  {
  }

  template <typename Act_t, typename Aggr_t>
  void addLayer(LayerParam const& param)
  {
    unsigned inputSize = 0;
    if(_layers.size() != 0)
    {
      inputSize = _layers[_layers.size()-1]->size();
    }
    _layers.push_back(std::make_shared<Layer<Act_t, Aggr_t>>(inputSize, param));
  }


protected:
  std::vector<std::shared_ptr<ILayer>> _layers;

  unsigned const _batchSize;
  double _learningRate;
  unsigned const _maxEpoch;
  unsigned const _epochAfterOptimal;

  std::vector<std::pair<std::vector<double>, std::vector<double>>> _trainData;
  std::vector<std::pair<std::vector<double>, std::vector<double>>> _validationData;
  std::vector<std::pair<std::vector<double>, std::vector<double>>> _testData;

  unsigned _epoch;
  unsigned _optimalEpoch;
};

}

#endif //BURNET_NETWORK_HH_