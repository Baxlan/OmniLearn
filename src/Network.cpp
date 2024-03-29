// Network.cpp

#include "omnilearn/cost.h"
#include "omnilearn/scheduler.h"
#include "omnilearn/Exception.hh"
#include "omnilearn/metric.h"
#include "omnilearn/Network.hh"



void omnilearn::Network::load(fs::path const& path, bool loadTestData, size_t threads)
{
  _pool = std::make_unique<ThreadPool>(threads);
  _param.threads = threads;
  Eigen::setNbThreads(static_cast<int>(threads));
  NetworkIO::load(*this, path, loadTestData);
}


void omnilearn::Network::addLayer(LayerParam const& param)
{
  if((param.aggregation == Aggregation::Distance || param.aggregation == Aggregation::Pdistance) &&
  (_param.dropconnect > std::numeric_limits<double>::epsilon() || _param.dropout > std::numeric_limits<double>::epsilon()))
    throw Exception("Dropout / dropconnect can't be used with layers having Distance / Pdistance aggregation function");
  _layers.push_back(Layer(param));
}


void omnilearn::Network::setParam(NetworkParam const& param)
{
  _param = param;

  if(_param.optimizer == Optimizer::Default)
  {
    _param.adaptiveLearningRate = false;
    _param.automaticLearningRate = false;
    _param.useMaxDenominator = false;
  }
  else if(_param.optimizer == Optimizer::Nadam)
  {
    _param.adaptiveLearningRate = true;
    _param.automaticLearningRate = false;
    _param.useMaxDenominator = false;
  }
  else if(_param.optimizer == Optimizer::AMSGrad)
  {
    _param.adaptiveLearningRate = true;
    _param.automaticLearningRate = false;
    _param.useMaxDenominator = true;
  }
  else if(_param.optimizer == Optimizer::Adadelta)
  {
    _param.adaptiveLearningRate = true;
    _param.automaticLearningRate = true;
    _param.useMaxDenominator = false;
  }
  else if(_param.optimizer == Optimizer::AdadeltaGrad)
  {
    _param.adaptiveLearningRate = true;
    _param.automaticLearningRate = true;
    _param.useMaxDenominator = true;
  }

  _param.seed = (_param.seed == 0 ? static_cast<unsigned>(std::chrono::steady_clock().now().time_since_epoch().count()) : _param.seed);
  _generator = std::mt19937(_param.seed);
  _dropoutDist = std::bernoulli_distribution(param.dropout);
  _dropconnectDist = std::bernoulli_distribution(param.dropconnect);
  _pool = std::make_unique<ThreadPool>(param.threads);
  Eigen::setNbThreads(static_cast<int>(param.threads));
}


void omnilearn::Network::setData(Data const& data)
{
  _trainInputs = data.inputs;
  _trainOutputs = data.outputs;
  _inputLabels = data.inputLabels;
  _outputLabels = data.outputLabels;
  _inputInfos = data.inputInfos;
  _outputInfos = data.outputInfos;
}


void omnilearn::Network::setTestData(Data const& data)
{
  _testInputs = data.inputs;
  _testOutputs = data.outputs;
}


void omnilearn::Network::learn()
{
  _io = std::make_unique<NetworkIO>(_param.name, _param.verbose);

  try
  {
    *_io << "\nChecking parameters\n";
    check();
    *_io << "Seed: " << _param.seed << "\n";
    *_io << "Shuffling and splitting data\n";
    splitData();
    *_io << "Preprocessing data...\n";
    initPreprocess();
    *_io << "Initializing layer and neuron parameters\n";
    _layers[_layers.size()-1].resize(static_cast<size_t>(_trainOutputs.cols()));
    initLayers();
    *_io << "Total number of trainable parameters: " << getNbParameters() << "\n";
    if (_param.loss == Loss::CrossEntropy || _param.loss == Loss::BinaryCrossEntropy)
    {
      *_io << "Initializing cross entropy weights\n\n";
      initCrossEntropyWeights();
    }
    *_io << "Training dataset size: " << _trainInputs.rows() << "\n";
    *_io << "Validation dataset size: " << _validationInputs.rows() << "\n";
    *_io << "Test dataset size: " << _testInputs.rows() << "\n\n";

    *_io << "Inputs: " << _trainInputs.cols() << " / " << _testInputs.cols() << " (" <<  _testInputs.cols() - _trainInputs.cols() << " discarded after reduction)\n";
    *_io << "Outputs: " << _trainOutputs.cols() << " / " << _testOutputs.cols() << " (" <<  _testOutputs.cols() - _trainOutputs.cols() << " discarded after reduction)\n";

    if(_param.loss == Loss::L1 || _param.loss == Loss::L2)
    {
      _testNormalizedOutputsForMetric = _testOutputs;
      _metricNormalization = normalize(_testNormalizedOutputsForMetric);
    }

    _iteration = 0;
    _broke = false;
    _firstTimeMaxBatchSizeReached = true;
    _iterationsSinceLastRestart = 0;
    _optimalEpoch = 0;
    keep();

    detectOptimalLearningRate();

    computeLoss();
    double lowestLoss = _validLosses[0];
    list(lowestLoss, Vector(0), true);

    for(_epoch = 1; _epoch < _param.epoch; _epoch++)
    {
      shuffleTrainData();
      Vector meanParameters = performeOneEpoch();
      computeLoss();

      double low = lowestLoss;

      if(1-(_validLosses[_epoch] / lowestLoss) > _param.improvement)
      {
        keep();
        lowestLoss = _validLosses[_epoch];
        _optimalEpoch = _epoch;
      }
      list(low, meanParameters, false);

      if(std::isnan(_trainLosses[_epoch]) || std::isnan(_validLosses[_epoch]) || std::isnan(_testMetric[_epoch]))
      // the 2nd and 4rd metric could be NaN without problem
        throw Exception("The last train, validation or test loss is NaN. The issue probably comes from too large weights.");

      //EARLY STOPPING
      if(_epoch - _optimalEpoch >= _param.patience)
        break;
    }
    release();
    *_io << "\nOptimal epoch: " << _optimalEpoch << "   First metric: " << _testMetric[_optimalEpoch] << "   Second metric: " << _testSecondMetric[_optimalEpoch] << "   Third metric: " << _testThirdMetric[_optimalEpoch] << "   Fourth metric: " << _testFourthMetric[_optimalEpoch] << "\n";
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
  inputs = preprocessIn(inputs);
  for(size_t i = 0; i < _layers.size(); i++)
  {
    inputs = _layers[i].process(inputs, *_pool);
  }
  // if cross-entropy loss is used, then score must be softmax
  if(_param.loss == Loss::CrossEntropy)
  {
    inputs = softmax(inputs);
  }
  inputs = dePreprocessOut(inputs);
  return inputs;
}


omnilearn::Vector omnilearn::Network::generate(NetworkParam param, Vector target, Vector input)
{
  if(input.size() == 0)
    input = Vector::Random(_layers[0].inputSize());
  else
    input = preprocessIn(input);
  target = preprocessOut(target);
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
  input = dePreprocessIn(input);
  return input;
}


omnilearn::Vector omnilearn::Network::preprocessIn(Vector inputs) const
{
  return preprocessIn(Matrix(inputs.transpose())).row(0);
}


omnilearn::Vector omnilearn::Network::preprocessOut(Vector outputs) const
{
  return preprocessOut(Matrix(outputs.transpose())).row(0);
}


omnilearn::Vector omnilearn::Network::dePreprocessIn(Vector inputs) const
{
  return dePreprocessIn(Matrix(inputs.transpose())).row(0);
}


omnilearn::Vector omnilearn::Network::dePreprocessOut(Vector outputs) const
{
  return dePreprocessOut(Matrix(outputs.transpose())).row(0);
}


omnilearn::Matrix omnilearn::Network::preprocessIn(Matrix inputs) const
{
  for(size_t i = 0; i < _param.preprocessInputs.size(); i++)
  {
    if(_param.preprocessInputs[i] == Preprocess::Normalize)
    {
      normalize(inputs, _inputNormalization);
    }
    else if(_param.preprocessInputs[i] == Preprocess::Standardize)
    {
      standardize(inputs, _inputStandartization);
    }
    else if(_param.preprocessInputs[i] == Preprocess::Whiten)
    {
      whiten(inputs, _param.inputWhiteningBias, _param.inputWhiteningType, _inputInfos, _inputDecorrelation);
    }
    else if(_param.preprocessInputs[i] == Preprocess::Reduce)
    {
      reduce(inputs, _inputDecorrelation, _param.inputReductionThreshold);
    }
  }
  return inputs;
}


omnilearn::Matrix omnilearn::Network::preprocessOut(Matrix outputs) const
{
  for(size_t i = 0; i < _param.preprocessOutputs.size(); i++)
  {
    if(_param.preprocessOutputs[i] == Preprocess::Normalize)
    {
      normalize(outputs, _outputNormalization);
    }
    if(_param.preprocessOutputs[i] == Preprocess::Standardize)
    {
      normalize(outputs, _outputStandartization);
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Whiten)
    {
      whiten(outputs, _param.outputWhiteningBias, _param.outputWhiteningType, _outputInfos, _outputDecorrelation);
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Reduce)
    {
      reduce(outputs, _outputDecorrelation, _param.outputReductionThreshold);
    }
  }
  return outputs;
}


omnilearn::Matrix omnilearn::Network::dePreprocessIn(Matrix inputs) const
{
  for(size_t pre = 0; pre < _param.preprocessInputs.size(); pre++)
  {
    if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Normalize)
    {
      deNormalize(inputs, _inputNormalization);
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Reduce)
    {
      deReduce(inputs, _inputDecorrelation);
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Whiten)
    {
      deWhiten(inputs, _param.inputWhiteningBias, _param.inputWhiteningType, _inputDecorrelation);
    }
    else if(_param.preprocessInputs[_param.preprocessInputs.size() - pre - 1] == Preprocess::Standardize)
    {
      deStandardize(inputs, _inputStandartization);
    }
  }
  return inputs;
}


omnilearn::Matrix omnilearn::Network::dePreprocessOut(Matrix outputs) const
{
  for(size_t pre = 0; pre < _param.preprocessOutputs.size(); pre++)
  {
    if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Normalize)
    {
      deNormalize(outputs, _outputNormalization);
    }
    else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Reduce)
    {
      deReduce(outputs, _outputDecorrelation);
    }
    else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Whiten)
    {
      deWhiten(outputs, _param.outputWhiteningBias, _param.outputWhiteningType, _outputDecorrelation);
    }
    else if(_param.preprocessOutputs[_param.preprocessOutputs.size() - pre - 1] == Preprocess::Standardize)
    {
      deStandardize(outputs, _outputStandartization);
    }
  }
  return outputs;
}


omnilearn::Data omnilearn::Network::getTestData() const
{
  Data data;

  data.inputLabels = _inputLabels;
  data.outputLabels = _outputLabels;
  data.inputs = _testInputs;
  data.outputs = _testOutputs;
  data.inputInfos = _inputInfos;
  data.outputInfos = _outputInfos;

  return data;
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
      _layers[i].init((i == 0 ? _trainInputs.cols() : _layers[i-1].size()), _generator);
  }
}


void omnilearn::Network::splitData()
{
  if(_testInputs.rows() != 0 && std::abs(_param.testRatio) > std::numeric_limits<double>::epsilon())
    throw Exception("TestRatio must be set to 0 because you already set a test dataset.");

  size_t validation = static_cast<size_t>(std::round(_param.validationRatio * static_cast<double>(_trainInputs.rows())));
  size_t test = static_cast<size_t>(std::round(_param.testRatio * static_cast<double>(_trainInputs.rows())));

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
  _trainInputs = Matrix(_trainInputs.topRows(_trainInputs.rows() - static_cast<eigen_size_t>(validation) - static_cast<eigen_size_t>(test)));
  _trainOutputs = Matrix(_trainOutputs.topRows(_trainOutputs.rows() - static_cast<eigen_size_t>(validation) - static_cast<eigen_size_t>(test)));


  // some checks about batch size
  size_t batchSize = static_cast<size_t>(_param.batchSizePercent ? _param.batchSize * static_cast<double>(_trainInputs.rows())/100 : _param.batchSize);
  size_t maxBatchSize = static_cast<size_t>(std::round(_param.maxBatchSizePercent ? _param.maxBatchSize * static_cast<double>(_trainInputs.rows())/100 : _param.maxBatchSize));

  if(batchSize < 1 || static_cast<eigen_size_t>(batchSize) > _trainInputs.rows())
    throw Exception("Batch size cannot be inferior to 1 nor greater than the training size.");

  if(batchSize > maxBatchSize)
    throw Exception("BatchBize cannot be greater than maxBatchSize.");

  if(_param.optimalLearningRateDetection && static_cast<eigen_size_t>(_param.learningRateSampling * batchSize) > _trainInputs.rows())
    throw Exception("Optimal learning rate detection is enabled : the number of samples times the batch size cannot be inferior to the training size");
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
  bool normalized = false;
  bool standardized = false;
  bool whitened = false;
  bool reduced = false;

  for(size_t i = 0; i < _param.preprocessInputs.size(); i++)
  {
    if(_param.preprocessInputs[i] == Preprocess::Normalize)
    {
      if(normalized == true)
        throw Exception("Inputs are normalized multiple times.");
      _inputNormalization = normalize(_trainInputs);
      normalize(_validationInputs, _inputNormalization);
      normalized = true;
    }
    else if(_param.preprocessInputs[i] == Preprocess::Standardize)
    {
      if(standardized == true)
        throw Exception("Inputs are standardized multiple times.");
      _inputStandartization = standardize(_trainInputs);
      standardize(_validationInputs, _inputStandartization);
      standardized = true;
    }
    else if(_param.preprocessInputs[i] == Preprocess::Whiten)
    {
      if(whitened == true)
        throw Exception("Inputs are whitened multiple times.");
      if(standardized == false)
        throw Exception("Inputs cannot be whitened without being standartized.");
      _inputDecorrelation = whiten(_trainInputs, _param.inputWhiteningBias, _param.inputWhiteningType, _inputInfos);
      whiten(_validationInputs, _param.inputWhiteningBias, _param.inputWhiteningType, _inputInfos, _inputDecorrelation);
      whitened = true;
    }
    else if(_param.preprocessInputs[i] == Preprocess::Reduce)
    {
      if(reduced == true)
        throw Exception("Inputs are reduced multiple times.");
      if(_param.inputWhiteningType == WhiteningType::ZCA)
        throw Exception("Inputs cannot be reduced after ZCA whitening. Try PCA whitening instead.");
      reduce(_trainInputs, _inputDecorrelation, _param.inputReductionThreshold);
      reduce(_validationInputs, _inputDecorrelation, _param.inputReductionThreshold);
      reduced = true;
    }
  }

  normalized = false;
  standardized = false;
  whitened = false;
  reduced = false;

  for(size_t i = 0; i < _param.preprocessOutputs.size(); i++)
  {
    if(_param.preprocessOutputs[i] == Preprocess::Whiten)
    {
      if(whitened == true)
        throw Exception("Outputs are whitened multiple times.");
      if(standardized == false)
        throw Exception("Outputs cannot be whitened without being standardized.");
      _outputDecorrelation = whiten(_trainOutputs, _param.outputWhiteningBias, _param.outputWhiteningType, _outputInfos);
      whiten(_validationOutputs, _param.outputWhiteningBias, _param.outputWhiteningType, _outputInfos, _outputDecorrelation);
      whitened = true;
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Reduce)
    {
      if(reduced == true)
        throw Exception("Outputs are reduced multiple times.");
      if(_param.outputWhiteningType == WhiteningType::ZCA)
        throw Exception("Outputs cannot be reduced after ZCA whitening. Try PCA whitening instead.");
      reduce(_trainOutputs, _outputDecorrelation, _param.outputReductionThreshold);
      reduce(_validationOutputs, _outputDecorrelation, _param.outputReductionThreshold);
      reduced = true;
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Normalize)
    {
      if(normalized == true)
        throw Exception("Outputs are normalized multiple times.");
      _outputNormalization = normalize(_trainOutputs);
      normalize(_validationOutputs, _outputNormalization);
      normalized = true;
    }
    else if(_param.preprocessOutputs[i] == Preprocess::Standardize)
    {
      if(standardized == true)
        throw Exception("Outputs are standardized multiple times.");
      _outputStandartization = standardize(_trainOutputs);
      standardize(_validationOutputs, _outputStandartization);
      standardized = true;
    }
  }
}


void omnilearn::Network::initCrossEntropyWeights()
{
  if(_param.weightMode == Weight::Disabled)
  {
  }
  else
  {
    if(_param.loss == Loss::L1 || _param.loss == Loss::L2)
    {
      throw Exception("Gradient weighting can only be used with (binary) cross entropy loss.");
    }
    if(_param.weightMode == Weight::Enabled)
    {
      if(_trainOutputs.cols() != _param.weights.size())
      {
        throw Exception("Weight vector must have the same size as the number of labels.");
      }
    }
    else if(_param.weightMode == Weight::Automatic)
    {
      _param.weights = Vector(_trainOutputs.cols());
      double power = std::log(0.5) / std::log(1/static_cast<double>(_trainOutputs.cols()));

      for(eigen_size_t i=0; i < _trainOutputs.cols(); i++)
      {
        _param.weights[i] = static_cast<double>(_trainOutputs.col(i).count())/static_cast<double>(_trainOutputs.rows());
        if(_param.loss == Loss::CrossEntropy)
          _param.weights[i] = std::pow(_param.weights[i], power);
      }
    }
  }
}


omnilearn::Vector omnilearn::Network::performeOneEpoch()
{
  std::bernoulli_distribution nullDist(0); // used to prevent the output neurons to get droppedOut

  std::vector<double> LR(0);
  std::vector<double> BS(0);
  std::vector<double> MO(0);

  size_t passedFeatures = 0;
  _scheduled = false;

  while(true) // loop exit condition is the break one
  {
    if(_broke)
      _broke = false;
    else
    {
      adaptLearningRate();
      adaptBatchSize();
      adaptMomentum();
    }

    if(_currentBatchSize > _trainInputs.rows() - passedFeatures)
    {
      _broke = true; // to avoid first iteration of next epoch to re-update LR, BS and momentum twice
      break;
    }

    _iteration += 1;
    LR.push_back(_currentLearningRate);
    BS.push_back(static_cast<double>(_currentBatchSize));
    MO.push_back(_currentMomentum);

    for(size_t feature = 0; feature < _currentBatchSize; feature++)
    {
      Vector featureInput = _trainInputs.row(passedFeatures);
      Vector featureOutput = _trainOutputs.row(passedFeatures);

      passedFeatures++;

      for(size_t i = 0; i < _layers.size(); i++)
      {
        featureInput = _layers[i].processToLearn(featureInput, (i == _layers.size()-1 ? nullDist : _dropoutDist), _dropconnectDist, _generator, *_pool);
      }

      Vector gradients(computeGradVector(featureOutput, featureInput));
      for(size_t i = 0; i < _layers.size(); i++)
      {
        _layers[_layers.size() - i - 1].computeGradients(gradients, *_pool);
        gradients = _layers[_layers.size() - i - 1].getGradients(*_pool);
      }
    }

    for(size_t i = 0; i < _layers.size(); i++)
    {
      _layers[i].updateWeights(_currentLearningRate, _param.L1, _param.L2, _param.decay, _param.automaticLearningRate, _param.adaptiveLearningRate, _param.useMaxDenominator, _currentMomentum, _previousMomentum, _nextMomentum, _cumulativeMomentum, _param.window, _param.optimizerBias, _iteration, *_pool);
    }
  }
  _missedData = _trainInputs.rows() - passedFeatures;

  Vector means(3);
  means(0) = stdToEigenVector(LR).mean();
  means(1) = stdToEigenVector(BS).mean();
  means(2) = stdToEigenVector(MO).mean();
  return means;
}


double omnilearn::Network::performeOptimalLearningRateDetection(std::vector<double>& LR, std::vector<double>& validationLoss, std::vector<double>& slopes)
{
  // get the different learning rates to be tested
  LR = std::vector<double>(0);
  for(size_t i = 0; i < _param.learningRateSampling; i++)
  {
    double tmp = static_cast<double>(i)/static_cast<double>(_param.learningRateSampling-1);
    tmp = _param.minLearningRate * std::exp(tmp*std::log(_param.learningRate/_param.minLearningRate));
    LR.push_back(tmp);
  }

  std::bernoulli_distribution nullDist(0); // used to prevent the output neurons to get droppedOut
  validationLoss = std::vector<double>(0);

  for(size_t batch = 0; batch < _param.learningRateSampling; batch++)
  {
    _iteration += 1;
    _currentLearningRate = LR[batch];
    for(size_t feature = 0; feature < _currentBatchSize; feature++)
    {
      Vector featureInput = _trainInputs.row(batch*_currentBatchSize + feature);
      Vector featureOutput = _trainOutputs.row(batch*_currentBatchSize + feature);

      for(size_t i = 0; i < _layers.size(); i++)
      {
        featureInput = _layers[i].processToLearn(featureInput, (i == _layers.size()-1 ? nullDist : _dropoutDist), _dropconnectDist, _generator, *_pool);
      }

      Vector gradients(computeGradVector(featureOutput, featureInput));
      for(size_t i = 0; i < _layers.size(); i++)
      {
        _layers[_layers.size() - i - 1].computeGradients(gradients, *_pool);
        gradients = _layers[_layers.size() - i - 1].getGradients(*_pool);
      }
    }

    for(size_t i = 0; i < _layers.size(); i++)
    {
      _layers[i].updateWeights(_currentLearningRate, _param.L1, _param.L2, _param.decay, _param.automaticLearningRate, _param.adaptiveLearningRate, _param.useMaxDenominator, _currentMomentum, _previousMomentum, _nextMomentum, _cumulativeMomentum, _param.window, _param.optimizerBias, _iteration, *_pool);
    }
    validationLoss.push_back(computeLossForOptimalLearningRateDetection());
  }

  // select the optimal learning rate
  slopes = std::vector<double>(0);
  double optimal = std::numeric_limits<double>::max();
  size_t optimalIndex = 0;
  size_t MA = _param.learningRateMovingAverage + 1;
  for(size_t i = MA; i < LR.size()-MA; i++)
  {
    slopes.push_back((validationLoss[i-MA] - validationLoss[i+MA]) / (std::log(LR[i-MA]) - std::log(LR[i+MA])));
    if(!std::isnan(slopes[slopes.size()-1]) && slopes[slopes.size()-1] < optimal)
    {
      optimal = slopes[slopes.size()-1];
      optimalIndex = i;
    }
  }

  if(optimalIndex == 0)
    throw Exception("No optimal leaning rate found, all result were probably NaN. This occurs when the learning rate range (sampling) is too high.");

  return LR[optimalIndex];
}


// process taking already processed inputs and giving processed outputs
// this allow to calculate loss without performing all the preprocessing
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
  bool useWeights = false;
  if(_param.weightMode == Weight::Enabled || _param.weightMode == Weight::Automatic)
    useWeights = true;

  if(_param.loss == Loss::L1)
    return L1Loss(realResult, predicted, *_pool);
  else if(_param.loss == Loss::L2)
    return L2Loss(realResult, predicted, *_pool);
  else if(_param.loss == Loss::BinaryCrossEntropy)
    return binaryCrossEntropyLoss(realResult, predicted, _param.crossEntropyBias, useWeights, _param.weights, *_pool);
  else if(_param.loss == Loss::CrossEntropy)
    return crossEntropyLoss(realResult, predicted, _param.crossEntropyBias, useWeights, _param.weights, *_pool);
  else
    throw Exception("Error while computing loss matrix, the loss function look ill-defined.");
}


omnilearn::Vector omnilearn::Network::computeGradVector(Vector const& realResult, Vector const& predicted) const
{
  bool useWeights = false;
  if(_param.weightMode == Weight::Enabled || _param.weightMode == Weight::Automatic)
    useWeights = true;

  if(_param.loss == Loss::L1)
    return L1Grad(realResult, predicted, *_pool);
  else if(_param.loss == Loss::L2)
    return L2Grad(realResult, predicted, *_pool);
  else if(_param.loss == Loss::BinaryCrossEntropy)
    return binaryCrossEntropyGrad(realResult, predicted, _param.crossEntropyBias, useWeights, _param.weights, *_pool);
  else //if loss == crossEntropy
    return crossEntropyGrad(realResult, predicted, useWeights, _param.weights, *_pool);
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
  double trainLoss = averageLoss(computeLossMatrix(_trainOutputs, processForLoss(input))) + L1 + L2;

  //validation loss
  double validationLoss = averageLoss(computeLossMatrix(_validationOutputs, processForLoss(_validationInputs))) + L1 + L2;

  //test metric
  std::array<double, 4> testMetric;
  if(_param.loss == Loss::L1 || _param.loss == Loss::L2)
    testMetric = regressionMetrics(_testNormalizedOutputsForMetric, process(_testInputs), _metricNormalization);
  else if(_param.loss == Loss::CrossEntropy || _param.loss == Loss::BinaryCrossEntropy)
    testMetric = classificationMetrics(_testOutputs, process(_testInputs), _param.classificationThreshold);

  _trainLosses.conservativeResize(_trainLosses.size() + 1);
  _trainLosses[_trainLosses.size()-1] = trainLoss;
  _validLosses.conservativeResize(_validLosses.size() + 1);
  _validLosses[_validLosses.size()-1] = validationLoss;
  _testMetric.conservativeResize(_testMetric.size() + 1);
  _testMetric[_testMetric.size()-1] = testMetric[0];
  _testSecondMetric.conservativeResize(_testSecondMetric.size() + 1);
  _testSecondMetric[_testSecondMetric.size()-1] = testMetric[1];
  _testThirdMetric.conservativeResize(_testThirdMetric.size() + 1);
  _testThirdMetric[_testThirdMetric.size()-1] = testMetric[2];
  _testFourthMetric.conservativeResize(_testFourthMetric.size() + 1);
  _testFourthMetric[_testFourthMetric.size()-1] = testMetric[3];
}


double omnilearn::Network::computeLossForOptimalLearningRateDetection() const
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

  return averageLoss(computeLossMatrix(_validationOutputs, processForLoss(_validationInputs))) + L1 + L2;
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
  size_t epochToUse = _epoch;
  size_t iterationToUse = _iteration;
  bool maxBatchSizeReached = (_currentBatchSize == static_cast<size_t>(std::round(_param.maxBatchSizePercent ? _param.maxBatchSize * static_cast<double>(_trainInputs.rows())/100 : _param.maxBatchSize)));

  if(_iteration == 0)
    _maxLearningRate = _param.learningRate;
  else if(_param.scheduleLearningRate)
  {
    if(_param.scheduleBatchSize)
    {
      if(maxBatchSizeReached)
      {
        epochToUse = _epoch - _epochWhenBatchSizeReachedMax;
        iterationToUse = _iteration - _iterationWhenBatchSizeReachedMax;
      }
      else
        return;
    }

    if(_param.scheduler == Scheduler::Exp)
      _maxLearningRate = LRexp(_param.learningRate, iterationToUse, _param.schedulerValue);
    else if(_param.scheduler == Scheduler::Step)
      _maxLearningRate = LRstep(_param.learningRate, iterationToUse, _param.schedulerValue, _param.schedulerDelay);
    else if(_param.scheduler == Scheduler::Plateau)
      if(epochToUse - _optimalEpoch > _param.schedulerDelay && !_scheduled)
      {
        _maxLearningRate /= _param.schedulerValue;
         _scheduled = true;
      }

    if(_maxLearningRate < _param.minLearningRate)
      _maxLearningRate = _param.minLearningRate;
  }

  if(_param.warmRestart && (!_param.scheduleBatchSize || (_param.scheduleBatchSize && maxBatchSizeReached)))
  {
    constexpr double pi = std::atan(1)*4;
    if(_iterationsSinceLastRestart == _param.warmRestartPeriod)
    {
      // this is warm restart
      _currentLearningRate = _maxLearningRate;
      _param.warmRestartPeriod = static_cast<size_t>(std::round(static_cast<double>(_param.warmRestartPeriod) * _param.warmRestartFactor));
      _iterationsSinceLastRestart = 0;
    }
    else
    {
      // this is cosine LR annealing
      _iterationsSinceLastRestart++;
      _currentLearningRate = _param.minLearningRate + 0.5*(_maxLearningRate - _param.minLearningRate)*(1+std::cos(static_cast<double>(_iterationsSinceLastRestart) * pi/ static_cast<double>(_param.warmRestartPeriod)));
    }
  }
  else
  {
    _currentLearningRate = _maxLearningRate;
  }
}


void omnilearn::Network::adaptBatchSize()
{
  size_t batchSize = static_cast<size_t>(_param.batchSizePercent ? _param.batchSize * static_cast<double>(_trainInputs.rows())/100 : _param.batchSize);
  size_t maxBatchSize = static_cast<size_t>(std::round(_param.maxBatchSizePercent ? _param.maxBatchSize * static_cast<double>(_trainInputs.rows())/100 : _param.maxBatchSize));

  if(_iteration == 0)
  {
    _currentBatchSize = batchSize;
    _epochWhenBatchSizeReachedMax = 1;
    _iterationWhenBatchSizeReachedMax = 1;
  }
  else if(_param.scheduleBatchSize && _currentBatchSize != maxBatchSize)
  {
    if(_param.scheduler == Scheduler::Exp)
      _currentBatchSize = static_cast<size_t>(std::round(BSexp(static_cast<double>(batchSize), _iteration, _param.schedulerValue)));
    else if(_param.scheduler == Scheduler::Step)
      _currentBatchSize = static_cast<size_t>(std::round(BSstep(static_cast<double>(batchSize), _iteration, _param.schedulerValue, _param.schedulerDelay)));
    else if(_param.scheduler == Scheduler::Plateau)
      if(_epoch - _optimalEpoch > _param.schedulerDelay && !_scheduled)
      {
        _currentBatchSize = static_cast<size_t>(std::round(static_cast<double>(_currentBatchSize) * _param.schedulerValue));
        _scheduled = true;
      }

    if(_currentBatchSize >= maxBatchSize && _firstTimeMaxBatchSizeReached)
    {
      _currentBatchSize = maxBatchSize;
      _epochWhenBatchSizeReachedMax = _epoch;
      _iterationWhenBatchSizeReachedMax = _iteration;
      _firstTimeMaxBatchSizeReached = false;
    }
  }
}


void omnilearn::Network::adaptMomentum()
{
  _previousMomentum = _currentMomentum;
  if(_param.momentumScheduler == Scheduler::Exp)
  {
    _currentMomentum = Mexp(_param.momentum, _param.maxMomentum, _iteration, _param.momentumSchedulerValue);
    _nextMomentum = Mexp(_param.momentum, _param.maxMomentum, _iteration+1, _param.momentumSchedulerValue);
  }
  else if(_param.momentumScheduler == Scheduler::Step)
  {
    _currentMomentum = Mstep(_param.momentum, _param.maxMomentum, _iteration, _param.momentumSchedulerValue, _param.momentumSchedulerDelay);
    _nextMomentum = Mstep(_param.momentum, _param.maxMomentum, _iteration+1, _param.momentumSchedulerValue, _param.momentumSchedulerDelay);
  }
  else
  {
    _currentMomentum = _param.momentum;
    _nextMomentum = _param.momentum;
  }
  // there is no plateau growth because we wouldn't be able to predict _nextMomentum

  if(_iteration == 0)
    _cumulativeMomentum = _currentMomentum;
  else
    _cumulativeMomentum *= _currentMomentum;
}


void omnilearn::Network::detectOptimalLearningRate()
{
  if(_param.optimalLearningRateDetection)
  {
    *_io << "\nDetection of optimal learning rate\n";
    Network net;

    // copy done manually to avoid copying test which could be large
    net._param = _param;
    net._generator = _generator;
    net._dropoutDist = _dropoutDist;
    net._dropconnectDist = _dropconnectDist;

    net._layers = std::vector<Layer>(0);
    for(size_t i = 0; i < _layers.size(); i++)
    {
      net._layers.push_back(_layers[i].getCopyForOptimalLearningRateDetection());
    }

    net._pool = std::make_unique<ThreadPool>(_param.threads);
    net._trainInputs = _trainInputs;
    net._trainOutputs = _trainOutputs;
    net._validationInputs = _validationInputs;
    net._validationOutputs = _validationOutputs;
    net._epoch = _epoch;
    net._iteration = _iteration;

    net.adaptBatchSize(); // to set the right batch size
    net.adaptMomentum(); // to set the right momentums

    std::vector<double> LR;
    std::vector<double> loss;
    std::vector<double> slope;

    _param.learningRate = net.performeOptimalLearningRateDetection(LR, loss, slope);

    *_io << "\nLearning rate, val. loss, slope" << "\n";
    for(std::size_t i = 0; i < slope.size(); ++i)
    {
      *_io << LR[i+_param.learningRateMovingAverage+1] << " " << loss[i+_param.learningRateMovingAverage+1] << " " << slope[i] << "\n";
    }

    *_io << "\nOptimal LR is " << _param.learningRate << "\n\n";
  }
}


void omnilearn::Network::check() const
{
  if(_param.learningRate < _param.minLearningRate)
    throw Exception("Learning rate cannot be inferior to minimum learning rate.");

  if(_param.batchSizePercent && (_param.batchSize <= 0 || _param.batchSize > 100))
    throw Exception("When batch size is expressed as percentage, it must be in ]0, 100].");

  if(_param.automaticLearningRate && !_param.adaptiveLearningRate)
    throw Exception("Cannot use automatic learning rate without adaptive learning rate.");

  if(_param.useMaxDenominator && !_param.adaptiveLearningRate)
    throw Exception("Cannot use the denominator correction (useMaxDenominator) without adaptive learning rate.");

  if(_param.adaptiveLearningRate && _param.window < 0.9 && (_param.momentum > 0 || _param.momentumScheduler != Scheduler::None))
    throw Exception("When using adaptive learning rate, the window parameter must be superior to 0.9 (because of Nesterov approximation) unless momentum is 0 and momentumSheduler is None.");

  if(_param.L1 < 0 || _param.L2 < 0 || _param.decay < 0)
    throw Exception("L1 / L2 regularization and weight decay cannot be negative.");

  if(_param.dropconnect < 0 || _param.dropconnect >= 1 || _param.dropout < 0 || _param.dropout >= 1)
    throw Exception("Dropout and dropconnect must be in [0, 1[.");

  if(_param.dropconnect > 0 && _param.dropout > 0)
    throw Exception("Dropout and dropconnect can't be used simultaneously.");

  if(_param.window < 0 || _param.window >= 1)
    throw Exception("Window must be in [0, 1[.");

  if(_param.momentum < 0 || _param.momentum >= 1 || _param.maxMomentum < 0 || _param.maxMomentum >= 1)
    throw Exception("Momentum and maxMomentum must be in [0, 1[.");

  if(_param.momentum > _param.maxMomentum)
    throw Exception("Momentum cannot be superior to maxMomentum.");

  if(_param.momentumScheduler == Scheduler::Plateau)
    throw Exception("Momentum cannot use the plateau scheduler, the next momentum needed for Nesterov wouldn't be predictible.");

  if(_param.automaticLearningRate && _param.optimalLearningRateDetection)
    throw Exception("Optimal scheduler and automatic learning rate cannot be used at the same time.");

  if(_param.optimalLearningRateDetection && _param.learningRateSampling < 2*_param.learningRateMovingAverage + 4)
    throw Exception("When the optimal scheduler is used, the sheduler value must be equal or greater than 4 + (2 x learningRateMovingAverage).");

  if((_param.scheduler == Scheduler::Plateau || _param.scheduler == Scheduler::Step) && _param.schedulerValue <= 1)
    throw Exception("The scheduler value must be greater than 1 when the plateau or the step scheduler is used.");

  if(_param.scheduler == Scheduler::Exp && _param.schedulerValue <= 0)
    throw Exception("The scheduler value must be greater than 0 when the exponential scheduler is used.");

  if(_param.momentumScheduler == Scheduler::Step && _param.momentumSchedulerValue <= 1)
    throw Exception("The momentum scheduler value must be greater than 1 when the step scheduler is used.");

  if(_param.momentumScheduler == Scheduler::Exp && _param.momentumSchedulerValue <= 0)
    throw Exception("The momentum scheduler value must be greater than 0 when the exponential scheduler is used.");

  if(_param.schedulerValue <= 0 || _param.momentumSchedulerValue <= 0)
    throw Exception("The scheduler values must be strictly positive.");

  if(_param.classificationThreshold < 0 || _param.classificationThreshold >= 1)
    throw Exception("The classification threshold must be in [0, 1[.");

  if((_param.loss == Loss::BinaryCrossEntropy || _param.loss == Loss::CrossEntropy) && _param.preprocessOutputs.size() != 0)
    throw Exception("Outputs cannot be preprocessed when using (binary) cross-entropy loss.");
}


size_t omnilearn::Network::getNbParameters() const
{
  size_t p = 0;

  for(size_t i=0; i<_layers.size(); i++)
  {
    p += _layers[i].getNbParameters();
  }

  return p;
}


void omnilearn::Network::list(double lowestLoss, Vector meanParameters, bool initial) const
{
  std::ostringstream oss;
  std::string currentLearningRate;

  if(_currentLearningRate >= 1e-6 && _currentLearningRate < 1e3)
  {
    oss << std::fixed << std::setprecision(6) << _currentLearningRate;
    currentLearningRate = oss.str();

    //formating the learning rate (that must be e string because of the ? operator (see below))
    for(size_t i=currentLearningRate.size()-1; i > 0; i--)
      if(currentLearningRate[i] == '0')
        currentLearningRate[i] = ' ';
      else
        break;
  }
  else
  {
    oss << std::scientific << std::setprecision(4) << (meanParameters.size() > 0 ? meanParameters(0) : 0);
    currentLearningRate = oss.str();
  }


  if(initial || _epoch % 50 == 0)
  {
    if(_param.loss == Loss::BinaryCrossEntropy || _param.loss == Loss::CrossEntropy)
    {
      *_io << "\n|   Epoch   | Validation |  Training  | Improvement since | Overfitting | Accuracy |    Mean    |    Mean    |   Mean     |    Mean    |   Mean    |   Mean   |  Ignored  | Remaining |\n";
      *_io <<   "|           |    Loss    |    Loss    |   optimal epoch   |             |          |  positive  |  negative  |   Cohen    |   global   |   batch   | Momentum |   data    |   epochs  |\n";
      *_io <<   "|           |            |            |                   |             |          | likelihood | likelihood |   kappa    |     LR     |   size    |          |           |           |\n\n";
    }
    else
    {
      *_io << "\n|   Epoch   | Validation |  Training  | Improvement since | Overfitting |   MAE    |   RMSE    |     Mean    |  Mean cos  |    Mean    |   Mean    |   Mean   |  Ignored  | Remaining |\n";
      *_io <<   "|           |    Loss    |    Loss    |   optimal epoch   |             |          |           | correlation | similarity |   global   |   batch   | Momentum |   data    |   epochs  |\n";
      *_io <<   "|           |            |            |                   |             |          |           |             |            |     LR     |   size    |          |           |           |\n";
    }
  }
  if(initial)
  {
    *_io << "| " << std::setw(9) << "0*" << " |  " << std::setw(9) << _validLosses[0] << " |  " << std::setw(9) << _trainLosses[0];
    *_io << " |     " << "   -    " << " %    |  " << std::setw(8) << std::round(100*(_validLosses[0]-_trainLosses[0])/_validLosses[0] *1e3)/1e3 << " % | ";
    *_io << std::setw(8) << _testMetric[0] << " |  " << std::setw(9) << _testSecondMetric[0] << " | " << std::setw(8) << _testThirdMetric[0] << "   |  " << std::setw(8) << _testFourthMetric[0] << "  | ";
    *_io << std::setw(10) << "    -   " << " | " << std::setw(9) << "    -    " << " | " << std::setw(8) << "   -    " << " | " << std::setw(9) << "    -    " << " | " << std::setw(9) << _param.patience << " |\n";
  }
  else
  {
    double gap = 100 * (1-(_validLosses[_epoch] / lowestLoss));
    *_io << "| " << std::setw(9) << std::to_string(_epoch) + (_epoch==_optimalEpoch ? "*": "") << " |  " << std::setw(9) << _validLosses[_epoch] << " |  " << std::setw(9) << _trainLosses[_epoch];
    *_io << " |     " << std::setw(8) << std::round(gap*1e3)/1e3 << " %    |  " << std::setw(8) << std::round(100*(_validLosses[_epoch]-_trainLosses[_epoch])/_validLosses[_epoch] * 1e3)/1e3 << " % | ";
    *_io << std::setw(8) << _testMetric[_epoch] << " |  " << std::setw(9) << _testSecondMetric[_epoch] << " |  " << std::setw(8) << _testThirdMetric[_epoch] << "  |  " << std::setw(8) << _testFourthMetric[_epoch] << "  | ";
    *_io << std::setw(10) << (_param.automaticLearningRate ? "    -   " : currentLearningRate);
    *_io << " | " << std::setw(9) << meanParameters(1) << " | " << std::setw(8) << meanParameters(2) << " | " << std::setw(9) << _missedData << " | " << std::setw(9) << _optimalEpoch + _param.patience - _epoch << " |\n";
  }
}
