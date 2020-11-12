// Network.cpp

#include "omnilearn/cost.h"
#include "omnilearn/scheduler.h"
#include "omnilearn/Exception.hh"
#include "omnilearn/metric.h"
#include "omnilearn/Network.hh"



void omnilearn::Network::load(fs::path const& path, size_t threads)
{
  _pool = std::make_unique<ThreadPool>(threads);
  _param.threads = threads;
  NetworkIO::load(*this, path);
}


void omnilearn::Network::addLayer(LayerParam const& param)
{
  _layers.push_back(Layer(param));
}


void omnilearn::Network::setParam(NetworkParam const& param)
{
  _param = param;
  _param.seed = (_param.seed == 0 ? static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count()) : _param.seed);
  _generator = std::mt19937(_param.seed);
  _dropoutDist = std::bernoulli_distribution(param.dropout);
  _dropconnectDist = std::bernoulli_distribution(param.dropconnect);
  _pool = std::make_unique<ThreadPool>(param.threads);
}


void omnilearn::Network::setData(Data const& data)
{
  _trainInputs = data.inputs;
  _trainOutputs = data.outputs;
  _inputLabels = data.inputLabels;
  _outputLabels = data.outputLabels;
}


void omnilearn::Network::setTestData(Data const& data)
{
  _testInputs = data.inputs;
  _testOutputs = data.outputs;

  //_testRawInputs = data.inputs;
  //_testRawOutputs = data.outputs;
  //not needed because they are set in the shuffle function
}


void omnilearn::Network::learn()
{
  _io = std::make_unique<NetworkIO>(_param.name, _param.verbose);

  try
  {
    *_io << "\nChecking parameters...\n";
    check();
    *_io << "Shuffling and splitting data...\n";
    splitData();
    *_io << "Preprocessing data...\n";
    initPreprocess();
    *_io << "Initializing layer and neuron parameters...\n";
    _layers[_layers.size()-1].resize(static_cast<size_t>(_trainOutputs.cols()));
    initLayers();

    *_io << "Inputs: " << _trainInputs.cols() << " / " << _testRawInputs.cols() << " (" <<  _testRawInputs.cols() - _trainInputs.cols() << " discarded after reduction)\n";
    *_io << "Outputs: " << _trainOutputs.cols() << " / " << _testRawOutputs.cols() << " (" <<  _testRawOutputs.cols() - _trainOutputs.cols() << " discarded after reduction)\n";

    _testNormalizedOutputsForMetric = _testRawOutputs;
    _metricNormalization = normalize(_testNormalizedOutputsForMetric);

    computeLoss();
    double lowestLoss = _validLosses[0];
    _optimalEpoch = 0;
    list(lowestLoss, true);
    keep();

    _iteration = 0;
    for(_epoch = 1; _epoch < _param.epoch; _epoch++)
    {
      shuffleTrainData();
      adaptLearningRate(); // sets _currentLearningRate
      adaptBatchSize(); // sets _currentBatchSize and _nbBatch
      adaptMomentum(); // sets _currentMomentum, _previousMomentum and _nextMomentum
      performeOneEpoch();
      computeLoss();

      double low = lowestLoss;
      //if validation loss is inferior to (_param.plateau * optimal loss), save current weights
      if(_validLosses[_epoch] < lowestLoss * _param.plateau)
      {
        keep();
        lowestLoss = _validLosses[_epoch];
        _optimalEpoch = _epoch;
      }
      list(low, false);

      if(std::isnan(_trainLosses[_epoch]) || std::isnan(_validLosses[_epoch]) || std::isnan(_testMetric[_epoch]))
        throw Exception("The last train, validation or test loss is NaN. The issue probably comes from too large weights.");

      //EARLY STOPPING
      if(_epoch - _optimalEpoch >= _param.patience)
        break;
    }
    release();
    *_io << "\nOptimal epoch: " << _optimalEpoch << "   First metric: " << _testMetric[_optimalEpoch] << "   Second metric: " << _testSecondMetric[_optimalEpoch] << "\n";
    _io->save(*this);
  }
  catch(Exception const& e)
  {
    *_io << e.what() << "\n";
    _io.reset();
    throw;
  }
  _io.reset();
}


omnilearn::Vector omnilearn::Network::process(Vector inputs) const
{
  return process(Matrix(inputs.transpose())).row(0);
}


omnilearn::Matrix omnilearn::Network::process(Matrix inputs) const
{
  inputs = preprocess(inputs);
  for(size_t i = 0; i < _layers.size(); i++)
  {
    inputs = _layers[i].process(inputs, *_pool);
  }
  // if cross-entropy loss is used, then score must be softmax
  if(_param.loss == Loss::CrossEntropy)
  {
    inputs = softmax(inputs);
  }
  inputs = postprocess(inputs);
  return inputs;
}


omnilearn::Vector omnilearn::Network::generate(NetworkParam param, Vector target, Vector input)
{
  if(input.size() == 0)
    input = Vector::Random(_layers[0].nbWeights());
  else
    input = preprocess(input);
  target = depostprocess(target);
  for(size_t iteration = 0; iteration < param.epoch; iteration++)
  {
    Vector res = input;
    for(size_t i = 0; i < _layers.size(); i++)
    {
      res = _layers[i].processToGenerate(res, *_pool);
    }
    Vector gradients(computeGradVector(target, res));
    for(size_t i = 0; i < _layers.size() - 1; i++)
    {
      _layers[_layers.size() - i - 1].computeGradients(gradients, *_pool);
      gradients = _layers[_layers.size() - i - 1].getGradients(*_pool);
    }
    _layers[0].computeGradientsAccordingToInputs(gradients, *_pool);
    _layers[0].updateInput(input, param.learningRate);

    for(size_t i = 0; i < _layers.size(); i++)
      _layers[i].resetGradientsForGeneration(*_pool);
  }
  input = depreprocess(input);
  return input;
}


omnilearn::Vector omnilearn::Network::preprocess(Vector inputs) const
{
  return preprocess(Matrix(inputs.transpose())).row(0);
}


omnilearn::Vector omnilearn::Network::postprocess(Vector outputs) const
{
  return postprocess(Matrix(outputs.transpose())).row(0);
}


omnilearn::Vector omnilearn::Network::depreprocess(Vector inputs) const
{
  return depreprocess(Matrix(inputs.transpose())).row(0);
}


omnilearn::Vector omnilearn::Network::depostprocess(Vector outputs) const
{
  return depostprocess(Matrix(outputs.transpose())).row(0);
}


omnilearn::Matrix omnilearn::Network::preprocess(Matrix inputs) const
{
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
    else if(_param.preprocessInputs[i] == Preprocess::Recorrelate)
    {
      for(eigen_size_t j = 0; j < inputs.rows(); j++)
      {
        inputs.row(j) = _inputDecorrelation.first * inputs.row(j).transpose();
      }
    }
  }
  return inputs;
}


omnilearn::Matrix omnilearn::Network::postprocess(Matrix outputs) const
{
  for(size_t pre = 0; pre < _param.preprocessOutputs.size(); pre++)
  {
    if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Normalize)
    {
      for(eigen_size_t i = 0; i < outputs.rows(); i++)
      {
        for(eigen_size_t j = 0; j < outputs.cols(); j++)
        {
          outputs(i,j) *= (_outputNormalization[j].second - _outputNormalization[j].first);
          outputs(i,j) += _outputNormalization[j].first;
        }
      }
    }
    else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Reduce)
    {
      Matrix newResults(outputs.rows(), _outputDecorrelation.second.size());
      rowVector zero = rowVector::Constant(_outputDecorrelation.second.size() - outputs.cols(), 0);
      for(eigen_size_t i = 0; i < outputs.rows(); i++)
      {
        newResults.row(i) = (rowVector(_outputDecorrelation.second.size()) << outputs.row(i), zero).finished();
      }
      outputs = newResults;
    }
    else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Decorrelate)
    {
      for(eigen_size_t i = 0; i < outputs.rows(); i++)
      {
        outputs.row(i) = _outputDecorrelation.first * outputs.row(i).transpose();
      }
    }
    else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Center)
    {
      for(eigen_size_t i = 0; i < outputs.rows(); i++)
      {
        for(eigen_size_t j = 0; j < outputs.cols(); j++)
        {
          outputs(i,j) += _outputCenter[j];
        }
      }
    }
  }
  return outputs;
}


omnilearn::Matrix omnilearn::Network::depreprocess(Matrix inputs) const
{
  for(size_t pre = 0; pre < _param.preprocessInputs.size(); pre++)
  {
    if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Normalize)
    {
      for(eigen_size_t i = 0; i < inputs.rows(); i++)
      {
        for(eigen_size_t j = 0; j < inputs.cols(); j++)
        {
          inputs(i,j) *= (_inputNormalization[j].second - _inputNormalization[j].first);
          inputs(i,j) += _inputNormalization[j].first;
        }
      }
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Reduce)
    {
      Matrix newResults(inputs.rows(), _inputDecorrelation.second.size());
      rowVector zero = rowVector::Constant(_inputDecorrelation.second.size() - inputs.cols(), 0);
      for(eigen_size_t i = 0; i < inputs.rows(); i++)
      {
        newResults.row(i) = (rowVector(_inputDecorrelation.second.size()) << inputs.row(i), zero).finished();
      }
      inputs = newResults;
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Decorrelate)
    {
      for(eigen_size_t i = 0; i < inputs.rows(); i++)
      {
        inputs.row(i) = _inputDecorrelation.first * inputs.row(i).transpose();
      }
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Center)
    {
      for(eigen_size_t i = 0; i < inputs.rows(); i++)
      {
        for(eigen_size_t j = 0; j < inputs.cols(); j++)
        {
          inputs(i,j) += _inputCenter[j];
        }
      }
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Standardize)
    {
      for(eigen_size_t i = 0; i < inputs.rows(); i++)
      {
        for(eigen_size_t j = 0; j < inputs.cols(); j++)
        {
          inputs(i,j) *= _inputStandartization[j].second;
          inputs(i,j) += _inputStandartization[j].first;
        }
      }
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Whiten)
    {
      for(eigen_size_t i = 0; i < inputs.cols(); i++)
      {
        inputs.col(i) *= (std::sqrt(_inputDecorrelation.second[i])+_param.inputWhiteningBias);
      }
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Recorrelate)
    {
      decorrelate(inputs, _inputDecorrelation);
    }
  }
  return inputs;
}


omnilearn::Matrix omnilearn::Network::depostprocess(Matrix outputs) const
{
  for(size_t i = 0; i < _param.preprocessOutputs.size(); i++)
  {
    if(_param.preprocessOutputs[i] == Preprocess::Center)
    {
      center(outputs, _outputCenter);
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Normalize)
    {
      normalize(outputs, _outputNormalization);
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Decorrelate)
    {
      decorrelate(outputs, _outputDecorrelation);
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Reduce)
    {
      reduce(outputs, _outputDecorrelation, _param.outputReductionThreshold);
    }
  }
  return outputs;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PRIVATE PART ============================================================
//=============================================================================
//=============================================================================
//=============================================================================



void omnilearn::Network::initLayers()
{
  for(size_t i = 0; i < _layers.size(); i++)
  {
      _layers[i].init((i == 0 ? _trainInputs.cols() : _layers[i-1].size()),
                      (i == _layers.size()-1 ? 0 : _layers[i+1].size()),
                      _generator);
  }
}


void omnilearn::Network::splitData()
{
  if(_testInputs.rows() != 0 && std::abs(_param.testRatio) > std::numeric_limits<double>::epsilon())
    throw Exception("TestRatio must be set to 0 because you already set a test dataset.");

  size_t validation = static_cast<size_t>(std::round(_param.validationRatio * static_cast<double>(_trainInputs.rows())));
  size_t test = static_cast<size_t>(std::round(_param.testRatio * static_cast<double>(_trainInputs.rows())));

  if(_param.batchSize > static_cast<size_t>(_trainInputs.rows()) - validation - test)
    throw Exception("The batch size is greater than the number of training data. Decrease the batch size, the validation ratio or the test ratio.");

  if(_param.batchSize == 0)
    _param.batchSize = static_cast<size_t>(_trainInputs.rows()) - validation - test; // if batch size == 0, then it is batch gradient descend

  shuffleTrainData();

  _validationInputs = Matrix(validation, _trainInputs.cols());
  _validationOutputs = Matrix(validation, _trainOutputs.cols());
  for(eigen_size_t i = 0; i < static_cast<eigen_size_t>(validation); i++)
  {
    _validationInputs.row(i) = _trainInputs.row(_trainInputs.rows()-1-i);
    _validationOutputs.row(i) = _trainOutputs.row(_trainOutputs.rows()-1-i);
  }
  if(_testInputs.rows() == 0)
  {
    _testInputs = Matrix(test, _trainInputs.cols());
    _testOutputs = Matrix(test, _trainOutputs.cols());
    for(eigen_size_t i = 0; i < static_cast<eigen_size_t>(test); i++)
    {
      _testInputs.row(i) = _trainInputs.row(_trainInputs.rows()-1-i-static_cast<eigen_size_t>(validation));
      _testOutputs.row(i) = _trainOutputs.row(_trainOutputs.rows()-1-i-static_cast<eigen_size_t>(validation));
    }
  }
  _testRawInputs = _testInputs; // testInputs will be preprocessed later
  _testRawOutputs = _testOutputs; // idem
  _trainInputs = Matrix(_trainInputs.topRows(_trainInputs.rows() - static_cast<eigen_size_t>(validation) - static_cast<eigen_size_t>(test)));
  _trainOutputs = Matrix(_trainOutputs.topRows(_trainOutputs.rows() - static_cast<eigen_size_t>(validation) - static_cast<eigen_size_t>(test)));
}


void omnilearn::Network::shuffleTrainData()
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


void omnilearn::Network::initPreprocess()
{
  bool centered = false;
  bool normalized = false;
  bool standardized = false;
  bool decorrelated = false;
  bool whitened = false;
  bool reduced = false;
  bool recorrelated = false;

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
      if(centered == false && standardized == false)
        throw Exception("Inputs cannot be decorrelated before centering or standartization.");
      _inputDecorrelation = decorrelate(_trainInputs);
      decorrelate(_validationInputs, _inputDecorrelation);
      decorrelate(_testInputs, _inputDecorrelation);
      decorrelated = true;
    }
    else if(_param.preprocessInputs[i] == Preprocess::Whiten)
    {
      if(whitened == true)
        throw Exception("Inputs are whitened multiple times.");
      if(decorrelated == false)
        throw Exception("Inputs cannot be whitened before decorrelation.");
      if(recorrelated == true)
        throw Exception("Inputs cannot be whitened after recorrelation.");
      whiten(_trainInputs, _inputDecorrelation, _param.inputWhiteningBias);
      whiten(_validationInputs, _inputDecorrelation, _param.inputWhiteningBias);
      whiten(_testInputs, _inputDecorrelation, _param.inputWhiteningBias);
      whitened = true;
    }
    else if(_param.preprocessInputs[i] == Preprocess::Reduce)
    {
      if(reduced == true)
        throw Exception("Inputs are reduced multiple times.");
      if(recorrelated == true)
        throw Exception("Inputs cannot be reduced after recorrelation.");
      reduce(_trainInputs, _inputDecorrelation, _param.inputReductionThreshold);
      reduce(_validationInputs, _inputDecorrelation, _param.inputReductionThreshold);
      reduce(_testInputs, _inputDecorrelation, _param.inputReductionThreshold);
      reduced = true;
    }
    else if(_param.preprocessInputs[i] == Preprocess::Recorrelate)
    {
      if(recorrelated == true)
        throw Exception("Inputs are recorrelated multiple times.");
      if(reduced == true)
        throw Exception("Inputs cannot be recorrelated after reduction.");

      for(eigen_size_t j = 0; j < _trainInputs.rows(); j++)
        _trainInputs.row(j) = _inputDecorrelation.first * _trainInputs.row(j).transpose();
      for(eigen_size_t j = 0; j < _validationInputs.rows(); j++)
        _validationInputs.row(j) = _inputDecorrelation.first * _validationInputs.row(j).transpose();
      for(eigen_size_t j = 0; j < _testInputs.rows(); j++)
        _testInputs.row(j) = _inputDecorrelation.first * _testInputs.row(j).transpose();

      recorrelated = true;
    }
  }

  centered = false;
  normalized = false;
  standardized = false;
  decorrelated = false;
  whitened = false;
  reduced = false;
  recorrelated = false;

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
      if(centered == false && standardized == false)
        throw Exception("Inputs cannot be decorrelated before centering.");
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
    else if(_param.preprocessOutputs[i] == Preprocess::Recorrelate)
    {
      throw Exception("Outputs can't be recorrelated.");
    }
  }
}


void omnilearn::Network::performeOneEpoch()
{
  for(size_t batch = 0; batch < _nbBatch; batch++)
  {
    _iteration += 1;
    for(size_t feature = 0; feature < _currentBatchSize; feature++)
    {
      Vector featureInput = _trainInputs.row(batch*_currentBatchSize + feature);
      Vector featureOutput = _trainOutputs.row(batch*_currentBatchSize + feature);

      for(size_t i = 0; i < _layers.size(); i++)
      {
        featureInput = _layers[i].processToLearn(featureInput, _param.dropout, _param.dropconnect, _dropoutDist, _dropconnectDist, _generator, *_pool);
      }

      Vector gradients(computeGradVector(featureOutput, featureInput));
      for(size_t i = 0; i < _layers.size(); i++)
      {
        _layers[_layers.size() - i - 1].computeGradients(gradients, *_pool);
        gradients = _layers[_layers.size() - i - 1].getGradients(*_pool);
      }
    }

    // because momentum is constant within an epoch, _nextMomentum is only valid at the last iteration of the epoch (it equals _currentMomentum otherwise)
    double mom = 0;
    if(batch != _nbBatch - 1)
      mom = _currentMomentum;
    else
      mom = _nextMomentum;

    if(_epoch == 1 && _iteration == 1)
      _cumulativeMomentum = _currentMomentum;
    else
      _cumulativeMomentum *= _currentMomentum;

    for(size_t i = 0; i < _layers.size(); i++)
    {
      _layers[i].updateWeights(_currentLearningRate, _param.L1, _param.L2, _param.decay, _param.automaticLearningRate, _param.adaptiveLearningRate, _currentMomentum, _previousMomentum, mom, _cumulativeMomentum, _param.window, _param.optimizerBias, _iteration, *_pool);
    }
    _previousMomentum = _currentMomentum; // because momentum is constant within an epoch, _previousMomentum was only valid at the first iteration of the epoch
  }
}


// process taking already processed inputs and giving processed outputs
// this allow to calculate loss and metrics without performing all the preprocessing
omnilearn::Matrix omnilearn::Network::processForLoss(Matrix inputs) const
{
  for(size_t i = 0; i < _layers.size(); i++)
  {
    inputs = _layers[i].process(inputs, *_pool);
  }
  // if cross-entropy loss is used, then score must be softmax
  if(_param.loss == Loss::CrossEntropy)
  {
    inputs = softmax(inputs);
  }
  return inputs;
}


omnilearn::Matrix omnilearn::Network::computeLossMatrix(Matrix const& realResult, Matrix const& predicted) const
{
  if(_param.loss == Loss::L1)
    return L1Loss(realResult, predicted, *_pool);
  else if(_param.loss == Loss::L2)
    return L2Loss(realResult, predicted, *_pool);
  else if(_param.loss == Loss::BinaryCrossEntropy)
    return binaryCrossEntropyLoss(realResult, predicted, *_pool);
  else //if loss == crossEntropy
    return crossEntropyLoss(realResult, predicted, *_pool);
}


omnilearn::Vector omnilearn::Network::computeGradVector(Vector const& realResult, Vector const& predicted) const
{
  if(_param.loss == Loss::L1)
    return L1Grad(realResult, predicted, *_pool);
  else if(_param.loss == Loss::L2)
    return L2Grad(realResult, predicted, *_pool);
  else if(_param.loss == Loss::BinaryCrossEntropy)
    return binaryCrossEntropyGrad(realResult, predicted, *_pool);
  else //if loss == crossEntropy
    return crossEntropyGrad(realResult, predicted, *_pool);
}


void omnilearn::Network::computeLoss()
{
  //L1 and L2 regularization loss
  double L1 = 0;
  double L2 = 0;
  std::pair<double, double> L1L2;

  for(size_t i = 0; i < _layers.size(); i++)
  //for each layer
  {
    L1L2 = _layers[i].L1L2(*_pool);
    L1 += L1L2.first;
    L2 += L1L2.second;
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
    testMetric = classificationMetrics(_testRawOutputs, process(_testRawInputs), _param.classificationThreshold);

  _trainLosses.conservativeResize(_trainLosses.size() + 1);
  _trainLosses[_trainLosses.size()-1] = trainLoss;
  _validLosses.conservativeResize(_validLosses.size() + 1);
  _validLosses[_validLosses.size()-1] = validationLoss;
  _testMetric.conservativeResize(_testMetric.size() + 1);
  _testMetric[_testMetric.size()-1] = testMetric.first;
  _testSecondMetric.conservativeResize(_testSecondMetric.size() + 1);
  _testSecondMetric[_testSecondMetric.size()-1] = testMetric.second;
}


void omnilearn::Network::keep()
{
    for(size_t i = 0; i < _layers.size(); i++)
    {
        _layers[i].keep();
    }
}


void omnilearn::Network::release()
{
    for(size_t i = 0; i < _layers.size(); i++)
    {
        _layers[i].release();
    }
}


void omnilearn::Network::adaptLearningRate()
{
  if(_epoch == 1)
    _currentLearningRate = _param.learningRate;
  else
  {
    size_t epochToUse = _epoch;
    if(_param.waitMaxBatchSize && _param.batchSizeScheduler != Scheduler::None)
    {
      if(_currentBatchSize != static_cast<size_t>(_trainInputs.rows()))
        return;
      else
        epochToUse = _epoch - _epochWhenBatchSizeReachedMax;
    }

    if(_param.learningRateScheduler == Scheduler::Inverse)
      _currentLearningRate = inverse(_param.learningRate, epochToUse, _param.learningRateSchedulerValue);
    else if(_param.learningRateScheduler == Scheduler::Exp)
      _currentLearningRate = exp(_param.learningRate, epochToUse, _param.learningRateSchedulerValue);
    else if(_param.learningRateScheduler == Scheduler::Step)
      _currentLearningRate = step(_param.learningRate, epochToUse, _param.learningRateSchedulerValue, _param.learningRateSchedulerDelay);
    else if(_param.learningRateScheduler == Scheduler::Plateau)
      if(epochToUse - _optimalEpoch > _param.learningRateSchedulerDelay)
          _currentLearningRate /= _param.learningRateSchedulerValue;
  }
}


void omnilearn::Network::adaptBatchSize()
{
  if(_epoch == 1)
  {
    _currentBatchSize = _param.batchSize;
    _epochWhenBatchSizeReachedMax = 1;
  }
  else if(_currentBatchSize != static_cast<size_t>(_trainInputs.rows()))
  {
    if(_param.batchSizeScheduler == Scheduler::Inverse)
      _currentBatchSize = static_cast<size_t>(std::round(growingInverse(static_cast<double>(_param.batchSize), static_cast<double>(_trainInputs.rows()), _epoch, _param.batchSizeSchedulerValue)));
    else if(_param.batchSizeScheduler == Scheduler::Exp)
      _currentBatchSize = static_cast<size_t>(std::round(growingExp(static_cast<double>(_param.batchSize), static_cast<double>(_trainInputs.rows()), _epoch, _param.batchSizeSchedulerValue)));
    else if(_param.batchSizeScheduler == Scheduler::Step)
      _currentBatchSize = static_cast<size_t>(std::round(growingStep(static_cast<double>(_param.batchSize), static_cast<double>(_trainInputs.rows()), _epoch, _param.batchSizeSchedulerValue, _param.batchSizeSchedulerDelay)));
    else if(_param.batchSizeScheduler == Scheduler::Plateau)
      if(_epoch - _optimalEpoch > _param.batchSizeSchedulerDelay)
      {
        size_t update = static_cast<size_t>(std::round((static_cast<double>(_trainInputs.rows()) - static_cast<double>(_currentBatchSize)) * _param.batchSizeSchedulerValue));
        if(update == 0)
          _currentBatchSize += 1;
        else
          _currentBatchSize += update;
      }
    _epochWhenBatchSizeReachedMax = _epoch;
  }
  _nbBatch = static_cast<size_t>(std::floor(_trainInputs.rows() / _currentBatchSize));
  _missedData = _trainInputs.rows() % _currentBatchSize;
}


void omnilearn::Network::adaptMomentum()
{
  _previousMomentum = _currentMomentum;
  if(_param.momentumScheduler == Scheduler::Inverse)
  {
    _currentMomentum = growingInverse(_param.momentum, _param.maxMomentum, _epoch, _param.momentumSchedulerValue);
    _nextMomentum = growingInverse(_param.momentum, _param.maxMomentum, _epoch+1, _param.momentumSchedulerValue);
  }
  else if(_param.momentumScheduler == Scheduler::Exp)
  {
    _currentMomentum = growingExp(_param.momentum, _param.maxMomentum, _epoch, _param.momentumSchedulerValue);
    _nextMomentum = growingExp(_param.momentum, _param.maxMomentum, _epoch+1, _param.momentumSchedulerValue);
  }
  else if(_param.momentumScheduler == Scheduler::Step)
  {
    _currentMomentum = growingStep(_param.momentum, _param.maxMomentum, _epoch, _param.momentumSchedulerValue, _param.momentumSchedulerDelay);
    _nextMomentum = growingStep(_param.momentum, _param.maxMomentum, _epoch+1, _param.momentumSchedulerValue, _param.momentumSchedulerDelay);
  }
  else
  {
    _currentMomentum = _param.momentum;
    _nextMomentum = _param.momentum;
  }
  // there is no plateau growth because we wouldn't be able to predict _nextMomentum
}


void omnilearn::Network::check() const
{
  if(_param.automaticLearningRate && !_param.adaptiveLearningRate)
    throw Exception("Cannot use automatic learning rate without adaptive learning rate.");

  if(_param.adaptiveLearningRate && _param.window < 0.9)
    throw Exception("When using adaptive learning rate, the window parameter must be superior to 0.9 (because of Nesterov approximation).");

  if(_param.L1 < 0 || _param.L2 < 0 || _param.decay < 0)
    throw Exception("L1 / L2 regularization and weight decay cannot be negative.");

  if(_param.dropconnect < 0 || _param.dropconnect >= 1 || _param.dropout < 0 || _param.dropout >= 1)
    throw Exception("Dropout and dropconnect must be in [0, 1[.");

  if(_param.window < 0 || _param.window >= 1)
    throw Exception("Window must be in [0, 1[.");

  if(_param.momentum < 0 || _param.momentum >= 1 || _param.maxMomentum < 0 || _param.maxMomentum >= 1)
    throw Exception("Momentum and maxMomentum must be in [0, 1[.");

  if(_param.momentum > _param.maxMomentum)
    throw Exception("Momentum cannot be superior to maxMomentum.");

  if(_param.momentumScheduler == Scheduler::Plateau)
    throw Exception("Momentum cannot use the plateau scheduler.");

  if(_param.batchSizeScheduler == Scheduler::Plateau && _param.batchSizeSchedulerValue >= 1)
    throw Exception("The batch size sheduler value must be in [0, 1[ when the plateau scheduler is used.");

  if(_param.learningRateSchedulerDelay <= 0 || _param.momentumSchedulerDelay <= 0 || _param.batchSizeSchedulerDelay <= 0)
    throw Exception("The different scheduler delays must be strictly positive.");

  if(_param.learningRateScheduler == Scheduler::Step && (_param.learningRateSchedulerValue < 0 || _param.learningRateSchedulerValue >=1))
    throw Exception("The learning rate scheduler value must be in [0, 1[ when the step scheduler is used.");

  if(_param.momentumScheduler == Scheduler::Step && (_param.momentumSchedulerValue < 0 || _param.momentumSchedulerValue >=1))
    throw Exception("The momentum scheduler value must be in [0, 1[ when the step scheduler is used.");

  if(_param.batchSizeScheduler == Scheduler::Step && (_param.batchSizeSchedulerValue < 0 || _param.batchSizeSchedulerValue >=1))
    throw Exception("The batch size scheduler value must be in [0, 1[ when the step scheduler is used.");
}


void omnilearn::Network::list(double lowestLoss, bool initial) const
{
  if(initial || _epoch % 50 == 0)
  {
    *_io << "\nEpoch       Validation     Training         current validation      Overfitting     First          Second         Global         Batch Size     Momentum       Remaining\n";
    *_io <<   "               Loss          Loss         compared to optimal one                   Metric         Metric      Learning rate                                     Epochs\n\n";
  }
  if(initial)
  {
    *_io << std::setw(9) << "0" << "   " << std::setw(12) << _validLosses[0] << "   " << std::setw(12) << _trainLosses[0];
    *_io << "     " << std::setw(12) << "0" << "%           " << std::setw(12) << (_validLosses[0]-_trainLosses[0])/_trainLosses[0] << "%   ";
    *_io << std::setw(12) << _testMetric[0] << "   " << std::setw(12) << _testSecondMetric[0] << "   ";
    *_io << std::setw(12) << "-" << "   " << std::setw(12) << "-" << "   " << std::setw(12) << "-" << "   " << std::setw(9) << _param.patience << "\n";
  }
  else
  {
    double gap = 100 * _validLosses[_epoch] / lowestLoss;
    *_io << std::setw(9) << _epoch << "   " << std::setw(12) << _validLosses[_epoch] << "   " << std::setw(12) << _trainLosses[_epoch];
    *_io << "     " << std::setw(12) << gap << "%           " << std::setw(12) << 100*(_validLosses[_epoch]-_trainLosses[_epoch])/_trainLosses[_epoch] << "%   ";
    *_io << std::setw(12) << _testMetric[_epoch] << "   " << std::setw(12) << _testSecondMetric[_epoch] << "   " << std::setw(12) << (_param.automaticLearningRate ? "-" : std::to_string(_currentLearningRate));
    *_io << "   " << std::setw(12) << _currentBatchSize << "   " << std::setw(12) << _currentMomentum << "   " << std::setw(9) << _optimalEpoch + _param.patience - _epoch << "   (" << _missedData << " data have been ignored)\n";
  }
}