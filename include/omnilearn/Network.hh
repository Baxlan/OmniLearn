// Network.hh

#ifndef OMNILEARN_NETWORK_HH_
#define OMNILEARN_NETWORK_HH_

#include "Layer.hh"
#include "cost.hh"
#include "decay.hh"
#include "metric.hh"
#include "csv.hh"
#include "fileString.hh"

#include <iostream>
#include <utility>



namespace omnilearn
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
    name("omnilearn_network")
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
  Network(Data const& data, NetworkParam const& param);
  Network(NetworkParam const& param, Data const& data);
  Network(std::string const& path, size_t threads);
  void addLayer(LayerParam const& param, size_t aggregation, size_t activation);
  void setTestData(Data const& data);
  bool learn();
  Matrix process(Matrix inputs) const;
  void writeInfo(std::string const& path) const;
  void saveNetInFile(std::string const& path) const;
  Vector generate(NetworkParam param, Vector target, Vector input = Vector(0));

protected:
  void initLayers();
  void shuffleTrainData();
  void shuffleData();
  void preprocess();
  void performeOneEpoch();
  //process taking already processed inputs and giving processed outputs
  Matrix processForLoss(Matrix inputs) const;
  Matrix computeLossMatrix(Matrix const& realResult, Matrix const& predicted);
  Vector computeGradVector(Vector const& realResult, Vector const& predicted);
  //return validation loss
  double computeLoss();
  void save();
  void loadSaved();

protected:
  //parameters
  NetworkParam _param;

  //random generators
  size_t _seed;
  std::mt19937 _generator;
  std::bernoulli_distribution _dropoutDist;
  std::bernoulli_distribution _dropconnectDist;

  //layers of neurons
  std::vector<Layer> _layers;

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



} // namespace omnilearn

#endif //OMNILEARN_NETWORK_HH_