#ifndef BRAIN_NETWORK_HH_
#define BRAIN_NETWORK_HH_

#include "Layer.hh"
#include "preprocess.hh"
#include "cost.hh"
#include "decay.hh"
#include "metric.hh"
#include "csv.hh"

#include <iostream>

namespace brain
{

enum class Loss {L1, L2, CrossEntropy, BinaryCrossEntropy};
enum class Metric {L1, L2, Accuracy};
enum class Preprocess {Center, Normalize, Standardize, Whiten, PCA};

//=============================================================================
//=============================================================================
//=============================================================================
//=== NETWORK PARAMETERS ======================================================
//=============================================================================
//=============================================================================
//=============================================================================


struct NetworkParam
{
    NetworkParam():
    seed(0),
    batchSize(0),
    learningRate(0.001),
    L1(0),
    L2(0),
    epoch(30),
    patience(5),
    dropout(0),
    dropconnect(0),
    validationRatio(0.2),
    testRatio(0.2),
    loss(Loss::L2),
    decayValue(0.05),
    decayDelay(5),
    decay(decay::none),
    classValidity(0.9),
    threads(1),
    optimizer(Optimizer::None),
    momentum(0.9),
    window(0.9),
    metric(Metric::L1),
    plateau(0.999),
    normalizeOutputs(false),
    preprocess()
    {
    }

    unsigned seed;
    unsigned batchSize;
    double learningRate;
    double L1;
    double L2;
    unsigned epoch;
    unsigned patience;
    double dropout;
    double dropconnect;
    double validationRatio;
    double testRatio;
    Loss loss;
    double decayValue;
    unsigned decayDelay;
    double (* decay)(double, unsigned, double, unsigned);
    double classValidity;
    unsigned threads;
    Optimizer optimizer;
    double momentum; //momentum
    double window; //window effect on grads
    Metric metric;
    double plateau;
    bool normalizeOutputs;
    std::vector<Preprocess> preprocess;
};



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
  Network(Data const& data, NetworkParam const& param):
  _param(param),
  _seed(param.seed == 0 ? static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count()) : param.seed),
  _generator(std::mt19937(_seed)),
  _dropoutDist(std::bernoulli_distribution(param.dropout)),
  _dropconnectDist(std::bernoulli_distribution(param.dropconnect)),
  _layers(),
  _pool(param.threads),
  _trainData(data.inputs),
  _trainRealResults(data.outputs),
  _validationData(0),
  _validationRealResults(0),
  _testData(0),
  _testRealResults(0),
  _testRawData(0),
  _nbBatch(0),
  _epoch(0),
  _optimalEpoch(0),
  _trainLosses(),
  _validLosses(),
  _testMetric(),
  _testSecondMetric(),
  _inputLabels(data.inputLabels),
  _outputLabels(data.outputLabels),
  _outputMinMax(),
  _centerData(),
  _normalizationData(),
  _standardizationData(),
  _whiteningData()
  {
  }


  Network(NetworkParam const& param, Data const& data):
  Network(data, param)
  {
  }

  template <typename Aggr_t = Dot, typename Act_t = Relu>
  void addLayer(LayerParam const& param = LayerParam())
  {
    _layers.push_back(std::make_shared<Layer<Aggr_t, Act_t>>(param));
  }


  void setTestData(Data const& data)
  {
    _testData = data.inputs;
    _testRealResults = data.outputs;
    _testRawData = data.inputs;
  }


  bool learn()
  {
    shuffleData();
    preprocess();
    initLayers();

    if(_param.normalizeOutputs)
    {
      _outputMinMax = normalize(_trainRealResults);
      normalize(_validationRealResults, _outputMinMax);
      normalize(_testRealResults, _outputMinMax);
    }
    else
    {
      _outputMinMax = std::vector<std::pair<double, double>>(_trainRealResults[0].size(), {0, 1});
    }

    if(_layers[_layers.size()-1]->size() != _trainRealResults[0].size())
    {
      throw Exception("The last layer must have as much neurons as outputs.");
    }

    double lowestLoss = computeLoss();
    std::cout << "\n";
    for(_epoch = 1; _epoch < _param.epoch; _epoch++)
    {
      performeOneEpoch();

      std::cout << "Epoch: " << _epoch;
      double validLoss = computeLoss();
      std::cout << "   LR: " << _param.decay(_param.learningRate, _epoch, _param.decayValue, _param.decayDelay) << "\n";
      if(std::isnan(_trainLosses[_epoch]) || std::isnan(validLoss))
        return false;

      //EARLY STOPPING
      if(validLoss < lowestLoss * _param.plateau) //if loss increases, or doesn't decrease more than 0.5% in _param.patience epochs, stop learning
      {
        save();
        lowestLoss = validLoss;
        _optimalEpoch = _epoch;
      }
      if(_epoch - _optimalEpoch > _param.patience)
        break;
    }
    loadSaved();
    std::cout << "\nOptimal epoch: " << _optimalEpoch << "   First metric: " << _testMetric[_optimalEpoch] << "   Second metric: " << _testSecondMetric[_optimalEpoch] << "\n";
    return true;
  }


  Matrix process(Matrix inputs) const
  {
    //preprocess inputs
    for(unsigned i = 0; i < _param.preprocess.size(); i++)
    {
      if(_param.preprocess[i] == Preprocess::Center)
      {
        center(inputs, _centerData);
      }
      else if(_param.preprocess[i] == Preprocess::Normalize)
      {
        normalize(inputs, _normalizationData);
      }
      else if(_param.preprocess[i] == Preprocess::Standardize)
      {
        standardize(inputs, _standardizationData);
      }
      else if(_param.preprocess[i] == Preprocess::Whiten)
      {
        whiten(inputs, _whiteningData.first);
      }
      else if(_param.preprocess[i] == Preprocess::PCA)
      {

      }
    }
    //process
    for(unsigned i = 0; i < _layers.size(); i++)
    {
      inputs = _layers[i]->process(inputs, _pool);
    }
    // if cross-entropy loss is used, then score must be softmax
    if(_param.loss == Loss::CrossEntropy)
    {
      inputs = softmax(inputs);
    }
    //denormalize outputs
    for(unsigned i = 0; i < inputs.lines(); i++)
    {
      for(unsigned j = 0; j < inputs.columns(); j++)
      {
        inputs[i][j] *= (_outputMinMax[j].second - _outputMinMax[j].first);
        inputs[i][j] += _outputMinMax[j].first;
      }
    }
    return inputs;
  }


  void writeInfo(std::string const& path) const
  {
    std::pair<Vector, Vector> acc;
    std::string loss;
    std::string metric;
    if(_param.metric == Metric::Accuracy)
      metric = "accuracy";
    else if(_param.metric == Metric::L1)
      metric = "mae";
    else if(_param.metric == Metric::L2)
      metric = "mse";

    if(_param.loss == Loss::BinaryCrossEntropy)
      loss = "binary cross entropy";
    else if(_param.loss == Loss::CrossEntropy)
      loss = "cross entropy";
    else if(_param.loss == Loss::L1)
      loss = "mae";
    else if(_param.loss == Loss::L2)
      loss = "mse";

    std::ofstream output(path);
    output << "labels:\n";
    for(unsigned i=0; i<_outputLabels.size(); i++)
    {
        output << _outputLabels[i] << ",";
    }
    output << "\n" << "loss:" << "\n" << loss << "\n";
    for(unsigned i=0; i<_trainLosses.size(); i++)
    {
        output << _trainLosses[i] << ",";
    }
    output << "\n";
    for(unsigned i=0; i<_validLosses.size(); i++)
    {
        output << _validLosses[i] << ",";
    }
    output << "\n" << "metric:" << "\n" << metric << "\n";
    for(unsigned i=0; i<_testMetric.size(); i++)
    {
        output << _testMetric[i] << ",";
    }
    output << "\n";
    for(unsigned i=0; i<_testSecondMetric.size(); i++)
    {
        output << _testSecondMetric[i] << ",";
    }
    if(_param.metric == Metric::Accuracy)
    {
      output << "\nthreshold:\n";
      output << _param.classValidity;
    }
    output << "\noptimal epoch:\n";
    output << _optimalEpoch;
    output << "\noutput normalization:\n";
    for(unsigned i=0; i<_outputMinMax.size(); i++)
    {
        output << _outputMinMax[i].first << ",";
    }
    output << "\n";
    for(unsigned i=0; i<_outputMinMax.size(); i++)
    {
        output << _outputMinMax[i].second << ",";
    }
    Matrix testRes(process(_testRawData));
    output << "\nexpected and predicted values:\n";
    for(unsigned i = 0; i < _outputLabels.size(); i++)
    {
      output << _outputLabels[i] << "\n" ;
      for(unsigned j = 0; j < _testRealResults.lines(); j++)
      {
        output << _testRealResults[j][i] << ",";
      }
      output << "\n";
      for(unsigned j = 0; j < testRes.lines(); j++)
      {
        output << testRes[j][i] << ",";
      }
    }
  }


protected:
  void initLayers()
  {
    for(unsigned i = 0; i < _layers.size(); i++)
    {
        _layers[i]->init((i == 0 ? _trainData.columns() : _layers[i-1]->size()),
                        (i == _layers.size()-1 ? _trainRealResults.columns() : _layers[i+1]->size()),
                        _param.batchSize, _generator);
    }
  }


  void shuffleData()
  {
    //shuffle inputs and outputs in the same order
    std::vector<size_t> indexes(_trainData.lines(), 0);
    for(size_t i = 0; i < indexes.size(); i++)
      indexes[i] = i;
    std::shuffle(indexes.begin(), indexes.end(), _generator);

    Matrix temp(_trainData.lines());
    for(size_t i = 0; i < indexes.size(); i++)
      temp[i] = _trainData[indexes[i]];
    std::swap(_trainData, temp);

    for(size_t i = 0; i < indexes.size(); i++)
      temp[i] = _trainRealResults[indexes[i]];
    std::swap(_trainRealResults, temp);

    if(_testData.lines() != 0 && std::abs(_param.testRatio) > std::numeric_limits<double>::epsilon())
      throw Exception("TestRatio must be set to 0 because you already set a test dataset.");

    double validation = _param.validationRatio * _trainData.lines();
    double test = _param.testRatio * _trainData.lines();
    double nbBatch = std::trunc(_trainData.lines() - validation - test) / _param.batchSize;

    //add a batch if an incomplete batch has more than 0.5*batchsize data
    if(nbBatch - static_cast<unsigned>(nbBatch) >= 0.5)
      nbBatch = std::trunc(nbBatch) + 1;

    unsigned nbTrain = static_cast<unsigned>(nbBatch)*_param.batchSize;
    unsigned noTrain = _trainData.lines() - nbTrain;
    validation = std::round(noTrain*_param.validationRatio/(_param.validationRatio + _param.testRatio));
    test = std::round(noTrain*_param.testRatio/(_param.validationRatio + _param.testRatio));

    for(unsigned i = 0; i < static_cast<unsigned>(validation); i++)
    {
      _validationData.addLine(_trainData[_trainData.lines()-1]);
      _validationRealResults.addLine(_trainRealResults[_trainRealResults.lines()-1]);
      _trainData.popLine();
      _trainRealResults.popLine();
    }
    for(unsigned i = 0; i < static_cast<unsigned>(test); i++)
    {
      _testData.addLine(_trainData[_trainData.lines()-1]);
      _testRealResults.addLine(_trainRealResults[_trainRealResults.lines()-1]);
      _testRawData.addLine(_trainData[_trainData.lines()-1]);
      _trainData.popLine();
      _trainRealResults.popLine();
    }
    _nbBatch = static_cast<unsigned>(nbBatch);
  }


  void preprocess()
  {
    for(unsigned i = 0; i < _param.preprocess.size(); i++)
    {
      if(_param.preprocess[i] == Preprocess::Center)
      {
        _centerData = center(_trainData);
        center(_validationData, _centerData);
        center(_testData, _centerData);
      }
      else if(_param.preprocess[i] == Preprocess::Normalize)
      {
        _normalizationData = normalize(_trainData);
        normalize(_validationData, _normalizationData);
        normalize(_testData, _normalizationData);
      }
      else if(_param.preprocess[i] == Preprocess::Standardize)
      {
        _standardizationData = standardize(_trainData);
        standardize(_validationData, _standardizationData);
        standardize(_testData, _standardizationData);
      }
      else if(_param.preprocess[i] == Preprocess::Whiten)
      {

      }
      else if(_param.preprocess[i] == Preprocess::PCA)
      {

      }
    }
  }


  void performeOneEpoch()
  {
    for(unsigned batch = 0; batch < _nbBatch; batch++)
    {
      Matrix input(_param.batchSize);
      Matrix output(_param.batchSize);

      std::vector<std::future<void>> tasks;
      for(unsigned i = 0; i < _param.batchSize; i++)
      {
        tasks.push_back(_pool.enqueue([this, &input, &output, i, batch]()->void
        {
          input[i] = _trainData[batch*_param.batchSize+i];
          output[i] = _trainRealResults[batch*_param.batchSize+i];
        }));
      }
      for(unsigned i = 0; i < tasks.size(); i++)
        tasks[i].get();

      for(unsigned i = 0; i < _layers.size(); i++)
      {
        input = _layers[i]->processToLearn(input, _param.dropout, _param.dropconnect, _dropoutDist, _dropconnectDist, _generator, _pool);
      }

      Matrix gradients(Matrix::transpose(computeLossMatrix(output, input).second));
      for(unsigned i = 0; i < _layers.size(); i++)
      {
        _layers[_layers.size() - i - 1]->computeGradients(gradients, _pool);
        gradients = _layers[_layers.size() - i - 1]->getGradients();
      }
      for(unsigned i = 0; i < _layers.size(); i++)
      {
        _layers[i]->updateWeights(_param.decay(_param.learningRate, _epoch, _param.decayValue, _param.decayDelay), _param.L1, _param.L2, _param.optimizer, _param.momentum, _param.window, _pool);
      }
    }
  }


  //process taking already processed inputs and giving normalized outputs
  Matrix processForLoss(Matrix inputs) const
  {
    for(unsigned i = 0; i < _layers.size(); i++)
    {
      inputs = _layers[i]->process(inputs, _pool);
    }
    // if cross-entropy loss is used, then score must be softmax
    if(_param.loss == Loss::CrossEntropy)
    {
      inputs = softmax(inputs);
    }
    return inputs;
  }


  std::pair<Matrix, Matrix> computeLossMatrix(Matrix const& realResults, Matrix const& predicted)
  {
    if(_param.loss == Loss::L1)
      return L1Loss(realResults, predicted);
    else if(_param.loss == Loss::L2)
      return L2Loss(realResults, predicted);
    else if(_param.loss == Loss::BinaryCrossEntropy)
      return binaryCrossEntropyLoss(realResults, predicted);
    else //if loss == crossEntropy
      return crossEntropyLoss(realResults, predicted);
  }


  //return validation loss
  double computeLoss()
  {
    //for each layer, for each neuron, first are weights, second are bias
    std::vector<std::vector<std::pair<Matrix, Vector>>> weights(_layers.size());
    for(unsigned i = 0; i < _layers.size(); i++)
    {
      weights[i] = _layers[i]->getWeights(_pool);
    }

    //L1 and L2 regularization loss
    double L1 = 0;
    double L2 = 0;

    for(unsigned i = 0; i < weights.size(); i++)
    //for each layer
    {
      for(unsigned j = 0; j < weights[i].size(); j++)
      //for each neuron
      {
        for(unsigned k = 0; k < weights[i][j].first.lines(); k++)
        //for each weight set
        {
          for(unsigned l = 0; l < weights[i][j].first[k].size(); l++)
          //for each weight
          {
            L1 += std::abs(weights[i][j].first[k][l]);
            L2 += std::pow(weights[i][j].first[k][l], 2);
          }
        }
      }
    }

    L1 *= _param.L1;
    L2 *= (_param.L2 * 0.5);

    //training loss
    Matrix input(_trainData.lines());
    Matrix output(_trainRealResults.lines());
    for(unsigned i = 0; i < _trainData.lines(); i++)
    {
      input[i] = _trainData[i];
      output[i] = _trainRealResults[i];
    }
    double trainLoss = averageLoss(computeLossMatrix(output, processForLoss(input)).first) + L1 + L2;

    //validation loss
    double validationLoss = averageLoss(computeLossMatrix(_validationRealResults, processForLoss(_validationData)).first) + L1 + L2;

    //test metric
    std::pair<double, double> testMetric;
    if(_param.metric == Metric::Accuracy)
      testMetric = accuracy(_testRealResults, processForLoss(_testData), _param.classValidity);
    else if(_param.metric == Metric::L1)
      testMetric = L1Metric(_testRealResults, processForLoss(_testData), _outputMinMax);
    else if(_param.metric == Metric::L2)
      testMetric = L2Metric(_testRealResults, processForLoss(_testData), _outputMinMax);


    std::cout << "   Valid_Loss: " << validationLoss << "   Train_Loss: " << trainLoss << "   First metric: " << (testMetric.first) << "   Second metric: " << (testMetric.second);
    _trainLosses.push_back(trainLoss);
    _validLosses.push_back(validationLoss);
    _testMetric.push_back(testMetric.first);
    _testSecondMetric.push_back(testMetric.second);
    return validationLoss;
  }


  void save()
  {
     for(unsigned i = 0; i < _layers.size(); i++)
      {
          _layers[i]->save();
      }
  }


  void loadSaved()
  {
     for(unsigned i = 0; i < _layers.size(); i++)
      {
          _layers[i]->loadSaved();
      }
  }


protected:
  NetworkParam _param;

  unsigned _seed;
  std::mt19937 _generator;
  std::bernoulli_distribution _dropoutDist;
  std::bernoulli_distribution _dropconnectDist;

  std::vector<std::shared_ptr<ILayer>> _layers;

  mutable ThreadPool _pool;

  Matrix _trainData;
  Matrix _trainRealResults;
  Matrix _validationData;
  Matrix _validationRealResults;
  Matrix _testData;
  Matrix _testRealResults;
  Matrix _testRawData;
  unsigned _nbBatch;

  unsigned _epoch;
  unsigned _optimalEpoch;
  Vector _trainLosses;
  Vector _validLosses;
  Vector _testMetric;
  Vector _testSecondMetric;

  std::vector<std::string> _inputLabels;
  std::vector<std::string> _outputLabels;
  std::vector<std::pair<double, double>> _outputMinMax;

  Vector _centerData;
  std::vector<std::pair<double, double>> _normalizationData;
  std::vector<std::pair<double, double>> _standardizationData;
  std::pair<Matrix, Vector> _whiteningData;
};



} // namespace brain

#endif //BRAIN_NETWORK_HH_