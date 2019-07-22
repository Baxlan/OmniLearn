#ifndef BURNET_NETWORK_HH_
#define BURNET_NETWORK_HH_

#include "Layer.hh"

#include <algorithm>
#include <iostream>

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
  Network(Dataset data = Dataset(), NetworkParam const& param = NetworkParam()):
  _dataSeed(param.dataSeed == 0 ? static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count()) : param.dataSeed),
  _dataGen(std::mt19937(_dataSeed)),
  _layers(),
  _batchSize(param.batchSize),
  _learningRate(param.learningRate),
  _L1(param.L1),
  _L2(param.L2),
  _tackOn(param.tackOn),
  _maxEpoch(param.maxEpoch),
  _epochAfterOptimal(param.epochAfterOptimal),
  _loss(param.loss),
  _validationRatio(param.validationRatio),
  _testRatio(param.testRatio),
  _trainData(data),
  _validationData(),
  _validationRealResults(),
  _testData(),
  _testRealResults(),
  _nbBatch(0),
  _epoch(0),
  _optimalEpoch(0)
  {
  }


  Network(NetworkParam const& param = NetworkParam(), Dataset data = Dataset()):
  Network(data, param)
  {
  }


  template <typename Aggr_t = Dot, typename Act_t = Relu>
  void addLayer(LayerParam const& param = LayerParam())
  {
    _layers.push_back(std::make_shared<Layer<Aggr_t, Act_t>>(param));
  }


  void setData(std::vector<std::pair<std::vector<double>, std::vector<double>>> data)
  {
    _trainData = data;
  }


  Matrix process(Matrix inputs) const
  {
    for(unsigned i = 0; i < _layers.size(); i++)
    {
      inputs = _layers[i]->process(inputs);
    }
    return inputs;
  }


  Matrix computeLossMatrix(Matrix const& realResults, Matrix const& predicted)
  {
    if(_loss == Loss::Cost)
      return cost(realResults, predicted);
    else if(_loss == Loss::SCost)
      return scost(realResults, predicted);
    else
      return Matrix();
  }


  // the inputs are loss, the output is average loss
  double averageLoss(Matrix const& loss)
  {
    std::vector<double> averageForEachFeature(loss.size());
    for(unsigned i = 0; i < loss.size(); i++)
    {
      for(unsigned j = 0; j < loss[0].size(); j++)
      {
        averageForEachFeature[i] += (loss[i][j]/loss[0].size());
      }
    }
    double average = 0;
    for(double i : averageForEachFeature)
      average += (i/averageForEachFeature.size());
    return average;
  }


 void learn()
  {
    initLayers();
    shuffleData();

    if(_layers[_layers.size()-1]->size() != _trainData[0].second.size())
    {
      throw Exception("The last layer must have as much neurons as outputs.");
    }

    for(;_epoch < _maxEpoch; _epoch++)
    {
      for(unsigned batch = 0; batch < _nbBatch; batch++)
      {
        Matrix input(_batchSize);
        Matrix output(_batchSize);
        for(unsigned i = 0; i < _batchSize; i++)
        {
          input[i] = _trainData[batch*_batchSize+i].first;
          output[i] = _trainData[batch*_batchSize+i].second;
        }

        for(unsigned i = 0; i < _layers.size(); i++)
        {
          input = _layers[i]->processToLearn(input);
        }

        Matrix loss(computeLossMatrix(output, input));
        Matrix gradients(transpose(loss));
        for(unsigned i = 0; i < _layers.size(); i++)
        {
          _layers[_layers.size() - i - 1]->computeGradients(gradients);
          gradients = _layers[_layers.size() - i - 1]->getGradients();
        }
        for(unsigned i = 0; i < _layers.size(); i++)
        {
          _layers[i]->updateWeights(_learningRate, _L1, _L2, _tackOn, 0);
        }
      }
      Matrix validationResult = process(_validationData);
      double validationLoss = averageLoss(computeLossMatrix(_validationRealResults, validationResult));
      std::cout << "Epoch: " << _epoch << "   Loss: " << validationLoss << "\n";
    }
  }


protected:
  void initLayers()
  {
    for(unsigned i = 0; i < _layers.size(); i++)
    {
        _layers[i]->init((i == 0 ? _trainData[0].first.size() : _layers[i-1]->size()),
                        (i == _layers.size()-1 ? _trainData[0].second.size() : _layers[i+1]->size()),
                        _batchSize);
    }
  }


  void shuffleData()
  {
    //testData(); //tests if all data have the same number of inputs and of output
    std::shuffle(_trainData.begin(), _trainData.end(), _dataGen);

    double validation = _validationRatio * _trainData.size();
    double test = _testRatio * _trainData.size();
    double nbBatch = std::trunc(_trainData.size() - validation - test) / _batchSize;

    //add a batch if an incomplete batch has more than 0.5*batchsize data
    if(nbBatch - static_cast<unsigned>(nbBatch) >= 0.5)
      nbBatch = std::trunc(nbBatch) + 1;

    unsigned nbTrain = static_cast<unsigned>(nbBatch)*_batchSize;
    unsigned noTrain = _trainData.size() - nbTrain;
    validation = std::round(noTrain*_validationRatio/(_validationRatio + _testRatio));
    test = std::round(noTrain*_testRatio/(_validationRatio + _testRatio));

    for(unsigned i = 0; i < static_cast<unsigned>(validation); i++)
    {
      _validationData.push_back(_trainData[_trainData.size()-1].first);
      _validationRealResults.push_back(_trainData[_trainData.size()-1].second);
      _trainData.pop_back();
    }
    for(unsigned i = 0; i < static_cast<unsigned>(test); i++)
    {
      _testData.push_back(_trainData[_trainData.size()-1].first);
      _testRealResults.push_back(_trainData[_trainData.size()-1].second);
      _trainData.pop_back();
    }
    _nbBatch = static_cast<unsigned>(nbBatch);
  }


protected:
  unsigned _dataSeed;
  std::mt19937 _dataGen;

  std::vector<std::shared_ptr<ILayer>> _layers;

  unsigned const _batchSize;
  double _learningRate;
  double _L1;
  double _L2;
  double _tackOn;
  unsigned const _maxEpoch;
  unsigned const _epochAfterOptimal;
  Loss _loss;

  double _validationRatio;
  double _testRatio;
  Dataset _trainData;
  Matrix _validationData;
  Matrix _validationRealResults;
  Matrix _testData;
  Matrix _testRealResults;
  unsigned _nbBatch;

  unsigned _epoch;
  unsigned _optimalEpoch;
};



} // namespace burnet

#endif //BURNET_NETWORK_HH_