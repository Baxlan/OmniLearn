#ifndef BRAIN_NETWORK_HH_
#define BRAIN_NETWORK_HH_

#include "Layer.hh"
#include "preprocess.hh"
#include "cost.hh"
#include "decay.hh"
#include "metric.hh"
#include "csv.hh"
#include "fileString.hh"

#include <iostream>
#include <utility>

namespace brain
{

enum class Loss {L1, L2, CrossEntropy, BinaryCrossEntropy};
enum class Metric {L1, L2, Accuracy};
enum class Preprocess {Center, Normalize, Standardize, Decorrelate, Whiten, Reduce};
enum class Decay {None, Inverse, Exp, Step, Plateau};

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
    epoch(100),
    patience(5),
    dropout(0),
    dropconnect(0),
    validationRatio(0.2),
    testRatio(0.2),
    loss(Loss::L2),
    decayValue(0.05),
    decayDelay(5),
    decay(Decay::None),
    classValidity(0.9),
    threads(1),
    optimizer(Optimizer::None),
    momentum(0.9),
    window(0.9),
    plateau(0.99),
    preprocessInputs(),
    preprocessOutputs(),
    optimizerBias(1e-5),
    inputReductionThreshold(0.99),
    outputReductionThreshold(0.99),
    inputWhiteningBias(1e-5),
    name("brain_network")
    {
    }

    size_t seed;
    size_t batchSize;
    double learningRate;
    double L1;
    double L2;
    size_t epoch;
    size_t patience;
    double dropout;
    double dropconnect;
    double validationRatio;
    double testRatio;
    Loss loss;
    double decayValue;
    size_t decayDelay;
    Decay decay;
    double classValidity;
    size_t threads;
    Optimizer optimizer;
    double momentum; //momentum
    double window; //window effect on grads
    double plateau;
    std::vector<Preprocess> preprocessInputs;
    std::vector<Preprocess> preprocessOutputs;
    double optimizerBias;
    double inputReductionThreshold;
    double outputReductionThreshold;
    double inputWhiteningBias;
    std::string name;
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
  _seed(param.seed == 0 ? static_cast<size_t>(std::chrono::steady_clock().now().time_since_epoch().count()) : param.seed),
  _generator(std::mt19937(_seed)),
  _dropoutDist(std::bernoulli_distribution(param.dropout)),
  _dropconnectDist(std::bernoulli_distribution(param.dropconnect)),
  _layers(),
  _pool(param.threads),
  _trainInputs(data.inputs),
  _trainOutputs(data.outputs),
  _validationInputs(),
  _validationOutputs(),
  _testInputs(),
  _testOutputs(),
  _testRawInputs(),
  _testRawOutputs(),
  _testNormalizedOutputsForMetric(),
  _nbBatch(),
  _epoch(),
  _optimalEpoch(),
  _trainLosses(),
  _validLosses(),
  _testMetric(),
  _testSecondMetric(),
  _inputLabels(data.inputLabels),
  _outputLabels(data.outputLabels),
  _outputCenter(),
  _outputNormalization(),
  _outputDecorrelation(),
  _metricNormalization(),
  _inputCenter(),
  _inputNormalization(),
  _inputStandartization(),
  _inputDecorrelation()
  {
  }


  Network(NetworkParam const& param, Data const& data):
  Network(data, param)
  {
  }


  Network(std::string const& path, size_t threads):
  _param(),
  _seed(),
  _generator(),
  _dropoutDist(),
  _dropconnectDist(),
  _layers(),
  _pool(threads),
  _trainInputs(),
  _trainOutputs(),
  _validationInputs(),
  _validationOutputs(),
  _testInputs(),
  _testOutputs(),
  _testRawInputs(),
  _testRawOutputs(),
  _testNormalizedOutputsForMetric(),
  _nbBatch(),
  _epoch(),
  _optimalEpoch(),
  _trainLosses(),
  _validLosses(),
  _testMetric(),
  _testSecondMetric(),
  _inputLabels(),
  _outputLabels(),
  _outputCenter(),
  _outputNormalization(),
  _outputDecorrelation(),
  _metricNormalization(),
  _inputCenter(),
  _inputNormalization(),
  _inputStandartization(),
  _inputDecorrelation()
  {
    std::vector<std::string> out = readCleanLines(path + ".out");
    std::vector<std::string> save = readCleanLines(path + ".save");
    std::string line;
    std::vector<std::string> vec;
    std::vector<std::string> vec2;

    // read .out to create param
    _param.threads = threads;

    for(size_t i = 0; i < out.size(); i++)
    {
      line = out[i];
      if(line == "loss:")
      {
        line = out[i+1];
        if(line == "mae")
          _param.loss = Loss::L1;
        if(line == "mse")
          _param.loss = Loss::L2;
        if(line == "binary cross entropy")
          _param.loss = Loss::BinaryCrossEntropy;
        if(line == "cross entropy")
          _param.loss = Loss::CrossEntropy;
      }
      else if(line == "input preprocess:")
      {
        vec = split(out[i+1], ',');
        for(std::string const& a : vec)
        {
          if(a == "center")
            _param.preprocessInputs.push_back(Preprocess::Center);
          if(a == "normalize")
            _param.preprocessInputs.push_back(Preprocess::Normalize);
          if(a == "standardize")
            _param.preprocessInputs.push_back(Preprocess::Standardize);
          if(a == "decorrelate")
            _param.preprocessInputs.push_back(Preprocess::Decorrelate);
          if(a == "whiten")
            _param.preprocessInputs.push_back(Preprocess::Whiten);
          if(a == "reduce")
            _param.preprocessInputs.push_back(Preprocess::Reduce);
        }
      }
      else if(line == "input eigenvalues:")
      {
        line = out[i+1];
        if(line != "0")
        {
          vec = split(line, ',');
          Vector eigenvalues(vec.size());
          for(size_t j = 0; j < vec.size(); j++)
          {
            eigenvalues[j] = std::stod(vec[j]);
          }
          _inputDecorrelation.second = eigenvalues;
          _param.inputReductionThreshold = std::stod(out[i+2]);
          _param.inputWhiteningBias = std::stod(out[i+3]);
        }
      }
      else if(line == "input eigenvectors:")
      {

      }
      else if(line == "input center:")
      {
        
      }
      else if(line == "input normalization:")
      {
        
      }
      else if(line == "input standardization:")
      {
        
      }
    }


    // read .save to load all weights / bias / coefs
    size_t inputSize = std::stoi(save[0]);
    for(size_t i = 1; i < save.size(); i++)
    {
      line = save[i];
      if(line.substr(0, 6) == "Layer: ")
      {
        size_t nbNeurons = std::stoi(line.erase(0, 6));
        i++;
        line = save[i];
        size_t aggreg = std::stoi(line.substr(0, line.find_first_of(" ")));
        size_t activ = std::stoi(line.substr(line.find_first_of(" ")+1, line.size()));
        addLayer(layerMap[{aggreg, activ}]());
        if(_layers.size() == 1)
          _layers[_layers.size()-1]->init(inputSize);
        else
          _layers[_layers.size()-1]->init(_layers[_layers.size()-2]->size());
        i++;
        for(size_t j = 0; j < nbNeurons; i++, j++)
        {
          vec = split(save[i], ' ');

          // load aggreg coefs
          size_t nbAggreg = std::stoi(vec[0]);
          Vector aggregation(nbAggreg);
          for(size_t k = 0; k < nbAggreg; k++)
          {
            aggregation[k] = std::stod(vec[k + 1]);
          }

          // load activ coefs
          size_t nbActiv = std::stoi(vec[nbAggreg + 1]);
          Vector activation(nbActiv);
          for(size_t k = 0; k < nbActiv; k++)
          {
            activation[k] = std::stod(vec[k + nbAggreg + 2]);
          }

          // load bias
          size_t nbBias = std::stoi(vec[nbAggreg + nbActiv + 2]);
          Vector bias(nbBias);
          for(size_t k = 0; k < nbBias; k++)
          {
            bias[k] = std::stod(vec[k + nbAggreg + nbActiv + 3]);
          }

          // load weights
          size_t nbWeights = std::stoi(vec[nbAggreg + nbActiv + nbBias + 3]);
          Vector weights(nbWeights);
          for(size_t k = 0; k < nbWeights; k++)
          {
            weights[k] = std::stod(vec[k + nbAggreg + nbActiv + nbBias + 4]);
          }

          // divide weights into wheight sets
          size_t weightsPerSet = nbWeights/nbBias;
          Matrix sets(weightsPerSet, nbBias);
          for(size_t k = 0; k < nbBias; k++)
          {
            for(size_t l = 0; l < weightsPerSet; l++)
            {
              sets(k, l) = weights[k*weightsPerSet + l];
            }
          }

          // put the coefs into the neuron
          _layers[_layers.size()-1]->setCoefs(j, sets, bias, aggregation, activation);
        }
      }
    }
  } // end of the loading constructor

  template <typename Aggr_t, typename Act_t>
  void addLayer(LayerParam const& param = LayerParam())
  {
    _layers.push_back(std::make_shared<Layer<Aggr_t, Act_t>>(param));
  }


  // non-const rvalue reference to only take temporary argument
  void addLayer(std::shared_ptr<ILayer>&& layer)
  {
    _layers.push_back(layer);
  }


  void setTestData(Data const& data)
  {
    _testInputs = data.inputs;
    _testOutputs = data.outputs;

    //_testRawInputs = data.inputs;
    //_testRawOutputs = data.outputs;
    //not needed because they are set in the shuffle function
  }


  bool learn()
  {
    shuffleData();
    preprocess();
    _layers[_layers.size()-1]->resize(static_cast<size_t>(_trainOutputs.cols()));
    initLayers();

    _testNormalizedOutputsForMetric = _testRawOutputs;
    _metricNormalization = normalize(_testNormalizedOutputsForMetric);

    std::cout << "inputs: " << _trainInputs.cols() << "/" << _testRawInputs.cols()<<"\n";
    std::cout << "outputs: " << _trainOutputs.cols() << "/" << _testRawOutputs.cols()<<"\n";

    double lowestLoss = computeLoss();
    std::cout << "\n";
    for(_epoch = 1; _epoch < _param.epoch; _epoch++)
    {
      performeOneEpoch();

      std::cout << "Epoch: " << _epoch;
      double validLoss = computeLoss();

      double lr = _param.learningRate;
      if(_param.decay == Decay::Inverse)
        lr = inverse(_param.learningRate, _epoch, _param.decayValue);
      else if(_param.decay == Decay::Exp)
        lr = exp(_param.learningRate, _epoch, _param.decayValue);
      else if(_param.decay == Decay::Step)
        lr = step(_param.learningRate, _epoch, _param.decayValue, _param.decayDelay);
      else if(_param.decay == Decay::Plateau)
        if(_epoch - _optimalEpoch > _param.decayDelay)
            _param.learningRate /= _param.decayValue;

      std::cout << "   LR: " << lr << "   gap from opti: " << 100 * validLoss / lowestLoss << "%   Remain. epochs: " << _optimalEpoch + _param.patience - _epoch + 1<< "\n";
      if(std::isnan(_trainLosses[_epoch]) || std::isnan(validLoss) || std::isnan(_testMetric[_epoch]))
        return false;

      //EARLY STOPPING
      if(validLoss < lowestLoss * _param.plateau) //if loss increases, or doesn't decrease more than _param.plateau percent in _param.patience epochs, stop learning
      {
        save();
        lowestLoss = validLoss;
        _optimalEpoch = _epoch;
      }
      if(_epoch - _optimalEpoch > _param.patience)
        break;

      //shuffle train data between each epoch
      shuffleTrainData();
    }
    loadSaved();
    std::cout << "\nOptimal epoch: " << _optimalEpoch << "   First metric: " << _testMetric[_optimalEpoch] << "   Second metric: " << _testSecondMetric[_optimalEpoch] << "\n";
    writeInfo(_param.name + ".out");
    saveNetInFile(_param.name + ".save");
    return true;
  }


  Matrix process(Matrix inputs) const
  {
    //preprocess inputs
    for(size_t i = 0; i < _param.preprocessInputs.size(); i++)
    {
      if(_param.preprocessInputs[i] == Preprocess::Center)
      {
        center(inputs, _inputCenter);
      }
      else if(_param.preprocessInputs[i] == Preprocess::Normalize)
      {
        normalize(inputs, _inputNormalization);
      }
      else if(_param.preprocessInputs[i] == Preprocess::Standardize)
      {
        standardize(inputs, _inputStandartization);
      }
      else if(_param.preprocessInputs[i] == Preprocess::Decorrelate)
      {
        decorrelate(inputs, _inputDecorrelation);
      }
      else if(_param.preprocessInputs[i] == Preprocess::Whiten)
      {
        whiten(inputs, _inputDecorrelation, _param.inputWhiteningBias);
      }
      else if(_param.preprocessInputs[i] == Preprocess::Reduce)
      {
        reduce(inputs, _inputDecorrelation, _param.inputReductionThreshold);
      }
    }
    //process
    for(size_t i = 0; i < _layers.size(); i++)
    {
      inputs = _layers[i]->process(inputs, _pool);
    }
    // if cross-entropy loss is used, then score must be softmax
    if(_param.loss == Loss::CrossEntropy)
    {
      inputs = softmax(inputs);
    }
    //transform computed outputs to real values
    for(size_t pre = 0; pre < _param.preprocessOutputs.size(); pre++)
    {
      if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Normalize)
      {
        for(eigen_size_t i = 0; i < inputs.rows(); i++)
        {
          for(eigen_size_t j = 0; j < inputs.cols(); j++)
          {
            inputs(i,j) *= (_outputNormalization[j].second - _outputNormalization[j].first);
            inputs(i,j) += _outputNormalization[j].first;
          }
        }
      }
      else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Reduce)
      {
        Matrix newResults(inputs.rows(), _outputDecorrelation.second.size());
        rowVector zero = rowVector::Constant(_outputDecorrelation.second.size() - inputs.cols(), 0);
        for(eigen_size_t i = 0; i < inputs.rows(); i++)
        {
          newResults.row(i) = (rowVector(_outputDecorrelation.second.size()) << inputs.row(i), zero).finished();
        }
        inputs = newResults;
      }
      else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Decorrelate)
      {
        for(eigen_size_t i = 0; i < inputs.rows(); i++)
        {
          inputs.row(i) = _outputDecorrelation.first * inputs.row(i).transpose();
        }
      }
      else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Center)
      {
        for(eigen_size_t i = 0; i < inputs.rows(); i++)
        {
          for(eigen_size_t j = 0; j < inputs.cols(); j++)
          {
            inputs(i,j) += _outputCenter[j];
          }
        }
      }
    }
    return inputs;
  }


  void writeInfo(std::string const& path) const
  {
    std::string loss;
    if(_param.loss == Loss::BinaryCrossEntropy)
      loss = "binary cross entropy";
    else if(_param.loss == Loss::CrossEntropy)
      loss = "cross entropy";
    else if(_param.loss == Loss::L1)
      loss = "mae";
    else if(_param.loss == Loss::L2)
      loss = "mse";

    std::ofstream output(path);
    if(!output)
      throw Exception("Cannot open/create file " + path);

    output << "input labels:\n";
    for(size_t i=0; i<_inputLabels.size(); i++)
        output << _inputLabels[i] << ",";
    output << "\noutput labels:\n";
    for(size_t i=0; i<_outputLabels.size(); i++)
        output << _outputLabels[i] << ",";
    output << "\n" << "loss:" << "\n" << loss << "\n";
    for(eigen_size_t i=0; i<_trainLosses.size(); i++)
        output << _trainLosses[i] << ",";
    output << "\n";
    for(eigen_size_t i=0; i<_validLosses.size(); i++)
        output << _validLosses[i] << ",";
    output << "\n" << "metric:" << "\n";
    for(eigen_size_t i=0; i<_testMetric.size(); i++)
        output << _testMetric[i] << ",";
    output << "\n";
    for(eigen_size_t i=0; i<_testSecondMetric.size(); i++)
        output << _testSecondMetric[i] << ",";
    if(loss == "binary cross entropy" || loss == "cross entropy")
    {
      output << "\nclassification threshold:\n";
      output << _param.classValidity;
    }
    output << "\noptimal epoch:\n";
    output << _optimalEpoch << "\n";

    output << "input preprocess:\n";
    for(size_t i = 0; i < _param.preprocessInputs.size(); i++)
    {
      if(_param.preprocessInputs[i] == Preprocess::Center)
        output << "center,";
      else if(_param.preprocessInputs[i] == Preprocess::Normalize)
        output << "normalize,";
      else if(_param.preprocessInputs[i] == Preprocess::Standardize)
        output << "standardize,";
      else if(_param.preprocessInputs[i] == Preprocess::Decorrelate)
        output << "decorrelate,";
      else if(_param.preprocessInputs[i] == Preprocess::Whiten)
        output << "whiten,";
      else if(_param.preprocessInputs[i] == Preprocess::Reduce)
        output << "reduce,";
    }
    output << "\ninput eigenvalues:\n";
    if(_inputDecorrelation.second.size() == 0)
      output << 0;
    else
    {
      for(eigen_size_t i = 0; i < _inputDecorrelation.second.size(); i++)
        output << _inputDecorrelation.second[i] << ",";
      output << "\n" << _param.inputReductionThreshold << "\n";
      output << _param.inputWhiteningBias << "\n";
    }
    output << "input eigenvectors:\n";
    if(_inputDecorrelation.second.size() == 0)
      output << 0;
    else
    {
      Matrix vectors = _inputDecorrelation.first.transpose();
      for(eigen_size_t i = 0; i < _inputDecorrelation.first.rows(); i++)
      {
        for(eigen_size_t j = 0; j < _inputDecorrelation.first.cols(); j++)
          output << vectors(i, j) << ",";
      }
    }
    output << "\ninput center:\n";
    if(_inputCenter.size() == 0)
      output << 0;
    else
      for(eigen_size_t i=0; i<_inputCenter.size(); i++)
          output << _inputCenter[i] << ",";
    output << "\n";
    output << "input normalization:\n";
    if(_inputNormalization.size() == 0)
      output << 0;
    else
    {
      for(size_t i=0; i<_inputNormalization.size(); i++)
          output << _inputNormalization[i].first << ",";
      output << "\n";
      for(size_t i=0; i<_inputNormalization.size(); i++)
          output << _inputNormalization[i].second << ",";
    }
    output << "\ninput standardization:\n";
    if(_inputStandartization.size() == 0)
      output << 0;
    else
    {
      for(size_t i=0; i<_inputStandartization.size(); i++)
          output << _inputStandartization[i].first << ",";
      output << "\n";
      for(size_t i=0; i<_inputStandartization.size(); i++)
          output << _inputStandartization[i].second << ",";
    }

    output << "\noutput preprocess:\n";
    for(size_t i = 0; i < _param.preprocessOutputs.size(); i++)
    {
      if(_param.preprocessOutputs[i] == Preprocess::Center)
        output << "center,";
      else if(_param.preprocessOutputs[i] == Preprocess::Decorrelate)
        output << "decorrelate,";
      else if(_param.preprocessOutputs[i] == Preprocess::Reduce)
        output << "reduce,";
      else if(_param.preprocessOutputs[i] == Preprocess::Normalize)
        output << "normalize,";
    }
    output << "\noutput eigenvalues:\n";
    if(_outputDecorrelation.second.size() == 0)
      output << 0;
    else
    {
      for(eigen_size_t i = 0; i < _outputDecorrelation.second.size(); i++)
        output << _outputDecorrelation.second[i] << ",";
      output << "\n" << _param.outputReductionThreshold << "\n";
    }
    output << "output eigenvectors:\n";
    if(_outputDecorrelation.second.size() == 0)
      output << 0;
    else
    {
      Matrix vectors = _outputDecorrelation.first.transpose();
      for(eigen_size_t i = 0; i < _outputDecorrelation.first.rows(); i++)
      {
        for(eigen_size_t j = 0; j < _outputDecorrelation.first.cols(); j++)
          output << vectors(i, j) << ",";
      }
    }
    output << "\noutput center:\n";
    if(_outputCenter.size() == 0)
      output << 0;
    else
      for(eigen_size_t i=0; i<_outputCenter.size(); i++)
          output << _outputCenter[i] << ",";
    output << "\n";
    output << "output normalization:\n";
    if(_outputNormalization.size() == 0)
      output << 0;
    else
    {
      for(size_t i=0; i<_outputNormalization.size(); i++)
          output << _outputNormalization[i].first << ",";
      output << "\n";
      for(size_t i=0; i<_outputNormalization.size(); i++)
          output << _outputNormalization[i].second << ",";
    }

    Matrix testRes(process(_testRawInputs));
    output << "\nexpected and predicted values:\n";
    for(size_t i = 0; i < _outputLabels.size(); i++)
    {
      output << "label: " << _outputLabels[i] << "\n" ;
      for(eigen_size_t j = 0; j < _testRawOutputs.rows(); j++)
        output << _testRawOutputs(j,i) << ",";
      output << "\n";
      for(eigen_size_t j = 0; j < testRes.rows(); j++)
        output << testRes(j,i) << ",";
      output << "\n";
    }
  }


  void saveNetInFile(std::string const& path) const
  {
    std::ofstream output(path);
    if(!output)
      throw Exception("Cannot open/create file " + path);
    output << _layers.size() << "\n";
    for(size_t i = 0; i < _layers.size(); i++)
    {
      output << "Layer: " << _layers[i]->size() << "\n";
      std::vector<rowVector> coefs = _layers[i]->getCoefs();
      for(size_t j = 0; j < coefs.size(); j++)
        output << coefs[j] << "\n";
    }
  }


  Vector generate(NetworkParam param, Vector target, Vector input = Vector(0))
  {
    if(input.size() == 0)
    {

    }
    for(size_t iteration = 0; iteration < param.epoch; iteration++)
    {
      Vector res = input;
      for(size_t i = 0; i < _layers.size(); i++)
      {
        res = _layers[i]->processToLearn(res, param.dropout, param.dropconnect, _dropoutDist, _dropconnectDist, _generator, _pool);
      }

      Vector gradients(computeGradVector(target, res));
      for(size_t i = 0; i < _layers.size() - 1; i++)
      {
        _layers[_layers.size() - i - 1]->computeGradients(gradients, _pool);
        gradients = _layers[_layers.size() - i - 1]->getGradients(_pool);
      }
      _layers[0]->computeGradientsAccordingToInputs(gradients, _pool);
      gradients = _layers[0]->getGradients(_pool);
      //_layers[0]->updateInputs(input, gradients, _pool);
    }
  }


protected:
  void initLayers()
  {
    for(size_t i = 0; i < _layers.size(); i++)
    {
        _layers[i]->init((i == 0 ? _trainInputs.cols() : _layers[i-1]->size()),
                        (i == _layers.size()-1 ? 0 : _layers[i+1]->size()),
                        _generator);
    }
  }


  void shuffleTrainData()
  {
    //shuffle inputs and outputs in the same order
    std::vector<size_t> indexes(_trainInputs.rows(), 0);
    for(size_t i = 0; i < indexes.size(); i++)
      indexes[i] = i;
    std::shuffle(indexes.begin(), indexes.end(), _generator);

    Matrix temp = Matrix(_trainInputs.rows(), _trainInputs.cols());
    for(size_t i = 0; i < indexes.size(); i++)
      temp.row(i) = _trainInputs.row(indexes[i]);
    std::swap(_trainInputs, temp);

    temp = Matrix(_trainOutputs.rows(), _trainOutputs.cols());
    for(size_t i = 0; i < indexes.size(); i++)
      temp.row(i) = _trainOutputs.row(indexes[i]);
    std::swap(_trainOutputs, temp);
  }


  void shuffleData()
  {
    shuffleTrainData();

    if(_testInputs.rows() != 0 && std::abs(_param.testRatio) > std::numeric_limits<double>::epsilon())
      throw Exception("TestRatio must be set to 0 because you already set a test dataset.");

    double validation = _param.validationRatio * static_cast<double>(_trainInputs.rows());
    double test = _param.testRatio * static_cast<double>(_trainInputs.rows());
    double nbBatch = std::trunc(static_cast<double>(_trainInputs.rows()) - validation - test) / static_cast<double>(_param.batchSize);
    if(_param.batchSize == 0)
      nbBatch = 1; // if batch size == 0, then is batch gradient descend

    //add a batch if an incomplete batch has more than 0.5*batchsize data
    if(nbBatch - std::trunc(nbBatch) >= 0.5)
      nbBatch = std::trunc(nbBatch) + 1;

    size_t noTrain = _trainInputs.rows() - (static_cast<size_t>(nbBatch)*_param.batchSize);
    validation = std::round(static_cast<double>(noTrain)*_param.validationRatio/(_param.validationRatio + _param.testRatio));
    test = std::round(static_cast<double>(noTrain)*_param.testRatio/(_param.validationRatio + _param.testRatio));

    _validationInputs = Matrix(static_cast<size_t>(validation), _trainInputs.cols());
    _validationOutputs = Matrix(static_cast<size_t>(validation), _trainOutputs.cols());
    if(_testInputs.rows() == 0)
    {
      _testInputs = Matrix(static_cast<size_t>(test), _trainInputs.cols());
      _testOutputs = Matrix(static_cast<size_t>(test), _trainOutputs.cols());
    }
    for(size_t i = 0; i < static_cast<size_t>(validation); i++)
    {
      _validationInputs.row(i) = _trainInputs.row(_trainInputs.rows()-1-i);
      _validationOutputs.row(i) = _trainOutputs.row(_trainOutputs.rows()-1-i);
    }
    for(size_t i = 0; i < static_cast<size_t>(test); i++)
    {
      _testInputs.row(i) = _trainInputs.row(_trainInputs.rows()-1-i-static_cast<size_t>(validation));
      _testOutputs.row(i) = _trainOutputs.row(_trainOutputs.rows()-1-i-static_cast<size_t>(validation));
    }
    _testRawInputs = _testInputs;
    _testRawOutputs = _testOutputs;
    _trainInputs = Matrix(_trainInputs.topRows(_trainInputs.rows() - validation - test));
    _trainOutputs = Matrix(_trainOutputs.topRows(_trainOutputs.rows() - validation - test));
    _nbBatch = static_cast<size_t>(nbBatch);
  }


  void preprocess()
  {
    bool centered = false;
    bool normalized = false;
    bool standardized = false;
    bool decorrelated = false;
    bool whitened = false;
    bool reduced = false;

    for(size_t i = 0; i < _param.preprocessInputs.size(); i++)
    {
      if(_param.preprocessInputs[i] == Preprocess::Center)
      {
        if(centered == true)
          throw Exception("Inputs are centered multiple times.");
        _inputCenter = center(_trainInputs);
        center(_validationInputs, _inputCenter);
        center(_testInputs, _inputCenter);
        centered = true;
      }
      else if(_param.preprocessInputs[i] == Preprocess::Normalize)
      {
        if(normalized == true)
          throw Exception("Inputs are normalized multiple times.");
        _inputNormalization = normalize(_trainInputs);
        normalize(_validationInputs, _inputNormalization);
        normalize(_testInputs, _inputNormalization);
        normalized = true;
      }
      else if(_param.preprocessInputs[i] == Preprocess::Standardize)
      {
        if(standardized == true)
          throw Exception("Inputs are standardized multiple times.");
        _inputStandartization = standardize(_trainInputs);
        standardize(_validationInputs, _inputStandartization);
        standardize(_testInputs, _inputStandartization);
        standardized = true;
      }
      else if(_param.preprocessInputs[i] == Preprocess::Decorrelate)
      {
        if(decorrelated == true)
          throw Exception("Inputs are decorrelated multiple times.");
        _inputDecorrelation = decorrelate(_trainInputs);
        decorrelate(_validationInputs, _inputDecorrelation);
        decorrelate(_testInputs, _inputDecorrelation);
        decorrelated = true;
      }
      else if(_param.preprocessInputs[i] == Preprocess::Whiten)
      {
        if(whitened == true)
          throw Exception("Inputs are whitened multiple times.");
        whiten(_trainInputs, _inputDecorrelation, _param.inputWhiteningBias);
        whiten(_validationInputs, _inputDecorrelation, _param.inputWhiteningBias);
        whiten(_testInputs, _inputDecorrelation, _param.inputWhiteningBias);
        whitened = true;
      }
      else if(_param.preprocessInputs[i] == Preprocess::Reduce)
      {
        if(reduced == true)
          throw Exception("Inputs are reduced multiple times.");
        reduce(_trainInputs, _inputDecorrelation, _param.inputReductionThreshold);
        reduce(_validationInputs, _inputDecorrelation, _param.inputReductionThreshold);
        reduce(_testInputs, _inputDecorrelation, _param.inputReductionThreshold);
        reduced = true;
      }
    }

    centered = false;
    normalized = false;
    standardized = false;
    decorrelated = false;
    whitened = false;
    reduced = false;

    for(size_t i = 0; i < _param.preprocessOutputs.size(); i++)
    {
      if(_param.preprocessOutputs[i] == Preprocess::Center)
      {
        if(centered == true)
          throw Exception("Outputs are centered multiple times.");
        _outputCenter = center(_trainOutputs);
        center(_validationOutputs, _outputCenter);
        center(_testOutputs, _outputCenter);
        centered = true;
      }
      else if(_param.preprocessOutputs[i] == Preprocess::Decorrelate)
      {
        if(decorrelated == true)
          throw Exception("Outputs are decorrelated multiple times.");
        _outputDecorrelation = decorrelate(_trainOutputs);
        decorrelate(_validationOutputs, _outputDecorrelation);
        decorrelate(_testOutputs, _outputDecorrelation);
        decorrelated = true;
      }
      else if(_param.preprocessOutputs[i] == Preprocess::Reduce)
      {
        if(reduced == true)
          throw Exception("Outputs are reduced multiple times.");
        reduce(_trainOutputs, _outputDecorrelation, _param.outputReductionThreshold);
        reduce(_validationOutputs, _outputDecorrelation, _param.outputReductionThreshold);
        reduce(_testOutputs, _outputDecorrelation, _param.outputReductionThreshold);
        reduced = true;
      }
      else if(_param.preprocessOutputs[i] == Preprocess::Normalize)
      {
        if(normalized == true)
          throw Exception("Outputs are normalized multiple times.");
        _outputNormalization = normalize(_trainOutputs);
        normalize(_validationOutputs, _outputNormalization);
        normalize(_testOutputs, _outputNormalization);
        normalized = true;
      }
      else if(_param.preprocessOutputs[i] == Preprocess::Whiten)
      {
        throw Exception("Outputs can't be whitened.");
      }
      else if(_param.preprocessOutputs[i] == Preprocess::Standardize)
      {
        throw Exception("Outputs can't be standardized.");
      }
    }
  }


  void performeOneEpoch()
  {
    for(size_t batch = 0; batch < _nbBatch; batch++)
    {
      for(size_t feature = 0; feature < _param.batchSize; feature++)
      {
        Vector featureInput = _trainInputs.row(batch*_param.batchSize + feature);
        Vector featureOutput = _trainOutputs.row(batch*_param.batchSize + feature);

        for(size_t i = 0; i < _layers.size(); i++)
        {
          featureInput = _layers[i]->processToLearn(featureInput, _param.dropout, _param.dropconnect, _dropoutDist, _dropconnectDist, _generator, _pool);
        }

        Vector gradients(computeGradVector(featureOutput, featureInput));
        for(size_t i = 0; i < _layers.size(); i++)
        {
          _layers[_layers.size() - i - 1]->computeGradients(gradients, _pool);
          gradients = _layers[_layers.size() - i - 1]->getGradients(_pool);
        }
      }

      double lr = _param.learningRate;
      //plateau decay is taken into account in learn()
      if(_param.decay == Decay::Inverse)
        lr = inverse(_param.learningRate, _epoch, _param.decayValue);
      else if(_param.decay == Decay::Exp)
        lr = exp(_param.learningRate, _epoch, _param.decayValue);
      else if(_param.decay == Decay::Step)
        lr = step(_param.learningRate, _epoch, _param.decayValue, _param.decayDelay);

      for(size_t i = 0; i < _layers.size(); i++)
      {
        _layers[i]->updateWeights(lr, _param.L1, _param.L2, _param.optimizer, _param.momentum, _param.window, _param.optimizerBias, _pool);
      }
    }
  }


  //process taking already processed inputs and giving processed outputs
  Matrix processForLoss(Matrix inputs) const
  {
    for(size_t i = 0; i < _layers.size(); i++)
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


  Matrix computeLossMatrix(Matrix const& realResult, Matrix const& predicted)
  {
    if(_param.loss == Loss::L1)
      return L1Loss(realResult, predicted, _pool);
    else if(_param.loss == Loss::L2)
      return L2Loss(realResult, predicted, _pool);
    else if(_param.loss == Loss::BinaryCrossEntropy)
      return binaryCrossEntropyLoss(realResult, predicted, _pool);
    else //if loss == crossEntropy
      return crossEntropyLoss(realResult, predicted, _pool);
  }


  Vector computeGradVector(Vector const& realResult, Vector const& predicted)
  {
    if(_param.loss == Loss::L1)
      return L1Grad(realResult, predicted, _pool);
    else if(_param.loss == Loss::L2)
      return L2Grad(realResult, predicted, _pool);
    else if(_param.loss == Loss::BinaryCrossEntropy)
      return binaryCrossEntropyGrad(realResult, predicted, _pool);
    else //if loss == crossEntropy
      return crossEntropyGrad(realResult, predicted, _pool);
  }

  //return validation loss
  double computeLoss()
  {
    //for each layer, for each neuron, first are weights, second are bias
    std::vector<std::vector<std::pair<Matrix, Vector>>> weights(_layers.size());
    for(size_t i = 0; i < _layers.size(); i++)
    {
      weights[i] = _layers[i]->getWeights(_pool);
    }

    //L1 and L2 regularization loss
    double L1 = 0;
    double L2 = 0;

    for(size_t i = 0; i < weights.size(); i++)
    //for each layer
    {
      for(size_t j = 0; j < weights[i].size(); j++)
      //for each neuron
      {
        for(eigen_size_t k = 0; k < weights[i][j].first.rows(); k++)
        //for each weight set
        {
          for(eigen_size_t l = 0; l < weights[i][j].first.cols(); l++)
          //for each weight
          {
            L1 += std::abs(weights[i][j].first(k, l));
            L2 += std::pow(weights[i][j].first(k, l), 2);
          }
        }
      }
    }

    L1 *= _param.L1;
    L2 *= (_param.L2 * 0.5);

    //training loss
    Matrix input = _trainInputs;
    Matrix output = _trainOutputs;
    double trainLoss = averageLoss(computeLossMatrix(output, processForLoss(input))) + L1 + L2;

    //validation loss
    double validationLoss = averageLoss(computeLossMatrix(_validationOutputs, processForLoss(_validationInputs))) + L1 + L2;

    //test metric
    std::pair<double, double> testMetric;
    if(_param.loss == Loss::L1 || _param.loss == Loss::L2)
      testMetric = regressionMetrics(_testNormalizedOutputsForMetric, process(_testRawInputs), _metricNormalization);
    else
      testMetric = classificationMetrics(_testRawOutputs, process(_testRawInputs), _param.classValidity);

    std::cout << "   Valid_Loss: " << validationLoss << "   Train_Loss: " << trainLoss << "   First metric: " << (testMetric.first) << "   Second metric: " << (testMetric.second);
    _trainLosses.conservativeResize(_trainLosses.size() + 1);
    _trainLosses[_trainLosses.size()-1] = trainLoss;
    _validLosses.conservativeResize(_validLosses.size() + 1);
    _validLosses[_validLosses.size()-1] = validationLoss;
    _testMetric.conservativeResize(_testMetric.size() + 1);
    _testMetric[_testMetric.size()-1] = testMetric.first;
    _testSecondMetric.conservativeResize(_testSecondMetric.size() + 1);
    _testSecondMetric[_testSecondMetric.size()-1] = testMetric.second;
    return validationLoss;
  }


  void save()
  {
     for(size_t i = 0; i < _layers.size(); i++)
      {
          _layers[i]->save();
      }
  }


  void loadSaved()
  {
     for(size_t i = 0; i < _layers.size(); i++)
      {
          _layers[i]->loadSaved();
      }
  }


protected:
  //parameters
  NetworkParam _param;

  //random generators
  size_t _seed;
  std::mt19937 _generator;
  std::bernoulli_distribution _dropoutDist;
  std::bernoulli_distribution _dropconnectDist;

  //layers of neurons
  std::vector<std::shared_ptr<ILayer>> _layers;

  //threadpool for parallelization
  mutable ThreadPool _pool;

  //data
  Matrix _trainInputs;
  Matrix _trainOutputs;
  Matrix _validationInputs;
  Matrix _validationOutputs;
  Matrix _testInputs;
  Matrix _testOutputs;
  Matrix _testRawInputs;
  Matrix _testRawOutputs;
  Matrix _testNormalizedOutputsForMetric;

  //learning infos
  size_t _nbBatch;
  size_t _epoch;
  size_t _optimalEpoch;
  Vector _trainLosses;
  Vector _validLosses;
  Vector _testMetric;
  Vector _testSecondMetric;

  //labels
  std::vector<std::string> _inputLabels;
  std::vector<std::string> _outputLabels;

  //output preprocessing
  Vector _outputCenter;
  std::vector<std::pair<double, double>> _outputNormalization;
  std::pair<Matrix, Vector> _outputDecorrelation;
  std::vector<std::pair<double, double>> _metricNormalization;

  //input preprocessing
  Vector _inputCenter;
  std::vector<std::pair<double, double>> _inputNormalization;
  std::vector<std::pair<double, double>> _inputStandartization;
  std::pair<Matrix, Vector> _inputDecorrelation;
};



} // namespace brain

#endif //BRAIN_NETWORK_HH_