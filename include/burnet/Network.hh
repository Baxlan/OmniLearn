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