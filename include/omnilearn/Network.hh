// Network.hh

#ifndef OMNILEARN_NETWORK_HH_
#define OMNILEARN_NETWORK_HH_

#include "NetworkIO.hh"
#include "Layer.hh"
#include "csv.h"

#include <filesystem>



namespace fs = std::filesystem;



namespace omnilearn
{

enum class Loss {L1, L2, CrossEntropy, BinaryCrossEntropy};
// In classification, if there are only two labels, one sigmoid output is enough (0 for one label, 1 for the other).
// In this case, BinaryCrossEntropy must be used. Indeed, using CrossEntropy would use softmax so the output would be 1 in all cases.
// If there are more than 2 labels, as many output neurons are needed. In this case CrossEntropy (linear outputs) and BinaryCrossEntropy (sigmoid) are similar.
enum class Preprocess {Normalize, Standardize, Whiten, Reduce};
enum class WhiteningType {PCA, ZCA};
enum class Scheduler {None, Exp, Step, Plateau};
enum class SecondOrder {None, Univariate, Multivariate};
enum class Weight {Disabled, Enabled, Automatic};
enum class Optimizer {None, Default, Nadam, AMSGrad ,Adadelta, AdadeltaGrad};
// All cases involve Nesterov and AdamW
// AdadeltaGrad = Adadelta combined with AMSgrad
// None: allows the user to set his own settings (3 bools)

// these bools are:

//                       | Default | Nadam | AMSGrad | Adadelta | AdadeltaGrad |
// adaptiveLearningRate  |    0    |    1  |    1    |    1     |      1       |
// automaticLearningRate |    0    |    0  |    0    |    1     |      1       |
// useMaxDenominator     |    0    |    0  |    1    |    0     |      1       |



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
    batchSize(1),
    batchSizePercent(false),
    scheduleLearningRate(false),
    scheduleBatchSize(false),
    maxBatchSize(100),
    maxBatchSizePercent(true),
    learningRate(0.01),
    minLearningRate(1e-7),
    L1(0),
    L2(0),
    decay(0), // weight decay
    epoch(1000),
    patience(5),
    dropout(0),
    dropconnect(0),
    validationRatio(0.2),
    testRatio(0.2),
    loss(Loss::L2),
    schedulerValue(2),
    schedulerDelay(2),
    optimalLearningRateDetection(false),
    learningRateMovingAverage(5),
    learningRateSampling(50),
    scheduler(Scheduler::None),
    classificationThreshold(0.5),
    threads(1),
    automaticLearningRate(false),
    adaptiveLearningRate(false),
    useMaxDenominator(false),
    momentum(0),
    maxMomentum(0.9),
    momentumSchedulerDelay(1),
    momentumSchedulerValue(1),
    momentumScheduler(Scheduler::None),
    window(0.99),
    improvement(0.01),
    preprocessInputs(),
    preprocessOutputs(),
    inputWhiteningType(WhiteningType::ZCA),
    outputWhiteningType(WhiteningType::ZCA),
    optimizerBias(1e-5),
    inputReductionThreshold(0.99),
    outputReductionThreshold(0.99),
    inputWhiteningBias(1e-5),
    outputWhiteningBias(1e-5),
    name("omnilearn_network"),
    verbose(false),
    optimizer(Optimizer::Default),
    crossEntropyBias(1e-3),
    weights(),
    weightMode(Weight::Disabled),
    warmRestart(false),
    warmRestartPeriod(20),
    warmRestartFactor(1)
    {
    }

    unsigned seed;
    double batchSize;
    bool batchSizePercent;
    bool scheduleLearningRate; // use the same scheduler, delay and value than BS ones
    bool scheduleBatchSize;    // use the same scheduler, delay and value than LR ones
    double maxBatchSize;
    bool maxBatchSizePercent;
    double learningRate;
    double minLearningRate;
    double L1;
    double L2;
    double decay;
    size_t epoch;
    size_t patience;
    double dropout;
    double dropconnect;
    double validationRatio;
    double testRatio;
    Loss loss;
    double schedulerValue;
    size_t schedulerDelay;
    bool optimalLearningRateDetection;
    size_t learningRateMovingAverage;
    size_t learningRateSampling;
    Scheduler scheduler;
    double classificationThreshold;
    size_t threads;
    bool automaticLearningRate;
    bool adaptiveLearningRate;
    bool useMaxDenominator;
    double momentum; //momentum
    double maxMomentum; //asymptotic value the momentum tries to reach in case of momentum shedule
    size_t momentumSchedulerDelay;
    double momentumSchedulerValue;
    Scheduler momentumScheduler;
    double window; //b2 in the second moment of gradients (and of updates)
    double improvement; // minimum validation loss improvement nedeed to become the new optimal
    std::vector<Preprocess> preprocessInputs;
    std::vector<Preprocess> preprocessOutputs;
    WhiteningType inputWhiteningType;
    WhiteningType outputWhiteningType;
    double optimizerBias;
    double inputReductionThreshold;
    double outputReductionThreshold;
    double inputWhiteningBias;
    double outputWhiteningBias;
    std::string name;
    bool verbose;
    Optimizer optimizer;
    double crossEntropyBias;
    Vector weights;
    Weight weightMode;
    bool warmRestart;
    size_t warmRestartPeriod;
    double warmRestartFactor;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== DECORRELATION DATA ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



struct DecorrelationData
{
    DecorrelationData():
    eigenVectors(0, 0),
    eigenValues(0),
    dummyScales(0),
    dummyMeans(0)
    {}

    Matrix eigenVectors;
    Vector eigenValues;
    Vector dummyScales;
    Vector dummyMeans;
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
  friend class NetworkIO; // NetworkIO is part of Network

public:
  Network() = default;
  void load(fs::path const& path, bool loadTestData, size_t threads);
  void addLayer(LayerParam const& param);
  void setParam(NetworkParam const& param);
  void setData(Data const& data);
  void setTestData(Data const& data);
  void learn();
  Vector process(Vector inputs) const;
  Matrix process(Matrix inputs) const;
  Vector generate(NetworkParam param, Vector target, Vector input = Vector(0));

  Vector preprocessIn(Vector inputs) const;
  Vector preprocessOut(Vector inputs) const;
  Vector dePreprocessIn(Vector inputs) const;
  Vector dePreprocessOut(Vector inputs) const;
  Matrix preprocessIn(Matrix inputs) const;
  Matrix preprocessOut(Matrix inputs) const;
  Matrix dePreprocessIn(Matrix inputs) const;
  Matrix dePreprocessOut(Matrix inputs) const;

  Data getTestData() const;

private:
  void initLayers();
  void splitData(); // shuffle data then split them into train/validation/test data
  void shuffleTrainData(); // shuffle train data each epoch
  void initPreprocess(); // first preprocess : calculate and store all the preprocessing data
  void initCrossEntropyWeights();
  Vector performeOneEpoch();
  double performeOptimalLearningRateDetection(std::vector<double>& LR, std::vector<double>& validationLoss, std::vector<double>& slopes);
  Matrix processForLoss(Matrix inputs) const; //takes preprocessed inputs, returns postprocessed outputs
  Matrix computeLossMatrix(Matrix const& realResult, Matrix const& predicted) const;
  Vector computeGradVector(Vector const& realResult, Vector const& predicted) const; // calculate error between expected and predicted outputs
  void computeLoss();
  double computeLossForOptimalLearningRateDetection() const;
  void keep(); // store weights, bias and other coefs when optimal loss is found
  void release(); // release weights, bias and other coefs when learning is done
  void adaptLearningRate();
  void adaptBatchSize();
  void adaptMomentum();
  void detectOptimalLearningRate();
  void check() const;
  size_t getNbParameters() const;
  void list(double lowestLoss, Vector meanParameters, bool initial) const;

private:
  //parameters
  NetworkParam _param;

  //random generators
  std::mt19937 _generator;
  std::bernoulli_distribution _dropoutDist;
  std::bernoulli_distribution _dropconnectDist;

  //layers of neurons
  std::vector<Layer> _layers;

  //threadpool for parallelization
  mutable std::unique_ptr<ThreadPool> _pool; // must be a pointer to be able to re-construct the pool (non copyable because of mutex)

  //data
  Matrix _trainInputs;
  Matrix _trainOutputs;
  Matrix _validationInputs;
  Matrix _validationOutputs;
  Matrix _testInputs;
  Matrix _testOutputs;
  Matrix _testNormalizedOutputsForMetric;

  //learning infos
  size_t _epoch;
  size_t _optimalEpoch;
  size_t _iteration;
  size_t _iterationsSinceLastRestart;
  double _currentLearningRate;
  double _maxLearningRate;
  double _currentMomentum;
  double _previousMomentum;
  double _nextMomentum;
  double _cumulativeMomentum;
  size_t _currentBatchSize;
  bool   _broke;
  bool   _firstTimeMaxBatchSizeReached;
  size_t _missedData; // # of data ignored because the minibatch would be incomplete
  size_t _epochWhenBatchSizeReachedMax;
  size_t _iterationWhenBatchSizeReachedMax;
  Vector _trainLosses;
  Vector _validLosses;
  Vector _testMetric;
  Vector _testSecondMetric;
  Vector _testThirdMetric;
  Vector _testFourthMetric;

  //labels
  std::vector<std::string> _inputLabels;
  std::vector<std::string> _outputLabels;

  //infos
  std::vector<std::string> _inputInfos;
  std::vector<std::string> _outputInfos;

  //output preprocessing
  std::vector<std::pair<double, double>> _outputNormalization;
  std::vector<std::pair<double, double>> _outputStandartization;
  DecorrelationData _outputDecorrelation;
  std::vector<std::pair<double, double>> _metricNormalization;

  //input preprocessing
  std::vector<std::pair<double, double>> _inputNormalization;
  std::vector<std::pair<double, double>> _inputStandartization;
  DecorrelationData _inputDecorrelation;

  //IO
  std::unique_ptr<NetworkIO> _io; // is a pointer because we only need IO during training
};



} // namespace omnilearn

#endif //OMNILEARN_NETWORK_HH_