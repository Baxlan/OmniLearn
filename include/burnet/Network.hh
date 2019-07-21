#ifndef BURNET_NETWORK_HH_
#define BURNET_NETWORK_HH_

#include "Layer.hh"

#include <algorithm>

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
  Network(std::vector<std::pair<std::vector<double>, std::vector<double>>> data = std::vector<std::pair<std::vector<double>, std::vector<double>>>(), NetworkParam const& param = NetworkParam()):
  _dataSeed(param.dataSeed == 0 ? static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count()) : param.dataSeed),
  _dataGen(std::mt19937(_dataSeed)),
  _layers(),
  _batchSize(param.batchSize),
  _learningRate(param.learningRate),
  _maxEpoch(param.maxEpoch),
  _epochAfterOptimal(param.epochAfterOptimal),
  _validationRatio(param.validationRatio),
  _testRatio(param.testRatio),
  _trainData(data),
  _validationData(),
  _testData(),
  _epoch(0),
  _optimalEpoch(0)
  {
  }


  template <typename Aggr_t = Dot, typename Act_t = Relu>
  void addLayer(LayerParam const& param = LayerParam())
  {
    _layers.push_back(std::make_shared<Layer<Aggr_t, Act_t>>(param));
  }


  void initLayers()
  {
    for(unsigned i = 0; i < _layers.size(); i++)
    {
        _layers[i]->init((i == 0 ? _trainData[0].first.size() : _layers[i-1]->size()),
                        (i == _layers.size()-1 ? _trainData[0].second.size() : _layers[i+1]->size()),
                        _batchSize);
    }
  }


  void setData(std::vector<std::pair<std::vector<double>, std::vector<double>>> data)
  {
    _trainData = data;
  }


  void shuffleData()
  {
    std::shuffle(_trainData.begin(), _trainData.end(), _dataGen);

    double validation = _validationRatio * _trainData.size();
    double test = _testRatio * _trainData.size();
    double nbBatch = std::trunc(_trainData.size() - validation - test) / _batchSize;

    //add a batch if an incomplete batch has more than 0.5*batchsize data
    if(nbBatch - static_cast<unsigned>(nbBatch) >= 0.5)
      nbBatch = std::trunc(nbBatch) + 1;

    unsigned nbTrain = static_cast<unsigned>(nbBatch)*_batchSize;
    unsigned noTrain = _trainData.size() - nbTrain;
    validation = std::round(noTrain*_validationRatio);

    _validationData.reserve(static_cast<unsigned>(validation));
    _testData.reserve(static_cast<unsigned>(test));

    test = std::round(noTrain*_testRatio);
    for(unsigned i = 0; i < static_cast<unsigned>(validation); i++)
    {
      _validationData[i] = _trainData[_trainData.size()-1];
      _trainData.pop_back();
    }
    for(unsigned i = 0; i < static_cast<unsigned>(test); i++)
    {
      _testData[i] = _trainData[_trainData.size()-1];
      _trainData.pop_back();
    }
  }


protected:
  unsigned _dataSeed;
  std::mt19937 _dataGen;

  std::vector<std::shared_ptr<ILayer>> _layers;

  unsigned const _batchSize;
  double _learningRate;
  unsigned const _maxEpoch;
  unsigned const _epochAfterOptimal;

  double _validationRatio;
  double _testRatio;
  std::vector<std::pair<std::vector<double>, std::vector<double>>> _trainData;
  std::vector<std::pair<std::vector<double>, std::vector<double>>> _validationData;
  std::vector<std::pair<std::vector<double>, std::vector<double>>> _testData;

  unsigned _epoch;
  unsigned _optimalEpoch;
};

}

#endif //BURNET_NETWORK_HH_