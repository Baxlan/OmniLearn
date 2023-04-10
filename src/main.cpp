// main.cpp

#include "omnilearn/Network.hh"
#include "omnilearn/metric.h"


void vesta()
{
    omnilearn::Data data = omnilearn::loadData("dataset/vesta.csv", ';', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 4;

    netp.validationRatio = 0.20;
    netp.testRatio = 0.20;
    netp.batchSize = 100;
    netp.maxBatchSize = 10;
    netp.maxBatchSizePercent = true;
    netp.scheduleBatchSize = true;
    netp.scheduleLearningRate = true;
    netp.learningRate = 0.1;
    netp.minLearningRate = 1e-5;
    netp.scheduler = omnilearn::Scheduler::Step;
    netp.schedulerValue = 1.5;
    netp.schedulerDelay = 1;

    netp.optimalLearningRateDetection = true;
    netp.learningRateMovingAverage = 5;
    netp.learningRateSampling = 100;

    netp.momentum = 0;
    //netp.maxMomentum = 0.90;
    //netp.momentumScheduler = omnilearn::Scheduler::Exp;
    //netp.momentumSchedulerValue = 0.1;

    netp.loss = omnilearn::Loss::L2;
    netp.patience = 10;
    netp.improvement = 0.01;

    netp.preprocessInputs = {omnilearn::Preprocess::Standardize, omnilearn::Preprocess::Whiten, omnilearn::Preprocess::Reduce};
    netp.preprocessOutputs = {omnilearn::Preprocess::Standardize, omnilearn::Preprocess::Whiten, omnilearn::Preprocess::Reduce, omnilearn::Preprocess::Normalize};
    netp.inputWhiteningType = omnilearn::WhiteningType::PCA;
    netp.outputWhiteningType = omnilearn::WhiteningType::PCA;

    netp.verbose = true;
    netp.optimizer = omnilearn::Optimizer::Default;
    netp.window = 0.99;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    lay.size = 24;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Relu;
    net.addLayer(lay);

    // output layer
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Linear;
    net.addLayer(lay);

    net.learn();
}


void iris()
{
    omnilearn::Data data = omnilearn::loadData("dataset/iris.csv", ',', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 1;
    netp.batchSize = 1;
    netp.learningRate = 0.01;

    netp.loss = omnilearn::Loss::CrossEntropy;
    netp.patience = 15;
    netp.improvement = 0.01;
    netp.momentum = 0.9;

    netp.classificationThreshold = 0.50;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.20;
    netp.preprocessInputs = {omnilearn::Preprocess::Standardize, omnilearn::Preprocess::Whiten, omnilearn::Preprocess::Reduce};
    netp.preprocessOutputs = {};
    netp.inputWhiteningType = omnilearn::WhiteningType::PCA;
    netp.inputReductionThreshold = 0.99;

    netp.verbose = true;
    netp.optimizer = omnilearn::Optimizer::Nadam;

    netp.dropout = 0.5;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    lay.size = 32;
    //lay.lockBias = true;
    //lay.lockWeights = true;
    //lay.lockParametric = true;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Softplus;
    net.addLayer(lay);

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Linear;
    net.addLayer(lay);

    net.learn();
}


void mnist()
{
    omnilearn::Data data = omnilearn::loadData("dataset/mnist_train.csv", ',', 4);
    omnilearn::Data testdata = omnilearn::loadData("dataset/mnist_test.csv", ',', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 4;

    netp.validationRatio = 0.20;
    netp.testRatio = 0.00;
    netp.batchSize = 1;
    netp.maxBatchSize = 0.1;
    netp.maxBatchSizePercent = true;
    netp.scheduleBatchSize = true;
    netp.scheduleLearningRate = true;
    netp.learningRate = 0.1;
    netp.scheduler = omnilearn::Scheduler::Plateau;
    netp.schedulerValue = 2;
    netp.schedulerDelay = 2;

    netp.momentum = 0;
    netp.maxMomentum = 0.90;
    netp.momentumScheduler = omnilearn::Scheduler::Exp;
    netp.momentumSchedulerValue = 0.2;

    netp.loss = omnilearn::Loss::CrossEntropy;
    netp.patience = 10;
    netp.improvement = 0.01;

    netp.classificationThreshold = 0.50;

    netp.preprocessInputs = {omnilearn::Preprocess::Standardize, omnilearn::Preprocess::Whiten, omnilearn::Preprocess::Reduce};
    netp.preprocessOutputs = {};
    netp.inputReductionThreshold = 0.90;

    netp.verbose = true;
    netp.optimizer = omnilearn::Optimizer::AMSGrad;
    netp.window = 0.99;



    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);
    net.setTestData(testdata);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    //lay.maxNorm = 100;
    lay.size = 128;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Prelu;
    net.addLayer(lay);

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Linear;
    net.addLayer(lay);

    net.learn();
}


void testLoader()
{
    omnilearn::Network genNet;
    genNet.load("omnilearn_network2", true, 4);
    //omnilearn::Data data = omnilearn::loadData("dataset/crypto_bot.csv", ';', 4);
    omnilearn::Data data = genNet.getTestData();
    std::array<double, 4> metric = omnilearn::classificationMetrics(data.outputs, genNet.process(data.inputs), 0.5);
    std::cout << metric[0] << " " << metric[1] << " " << metric[2] << " " << metric[3] << "\n";
}


void generate()
{
    std::srand(static_cast<unsigned>(time(0)));
    omnilearn::Network genNet;
    genNet.load("omnilearn_network", false, 4);
    omnilearn::NetworkParam param;
    param.epoch = 200;
    param.learningRate = 0.1;
    omnilearn::Vector target = (omnilearn::Vector(10) << 0,0,0,0,0,0,0,1,0,0).finished();
    omnilearn::Vector input = omnilearn::Vector::Random(28*28);

    for(eigen_size_t i = 0; i < input.size(); i++)
    {
        input[i] += 1;
        input[i] *= 127.5;
    }

    omnilearn::Vector res = genNet.generate(param, target, input);

    std::cout << res << "\n";
    std::cout << genNet.process(res) << "\n";
}



void cryptoBot()
{
    omnilearn::Data data = omnilearn::loadData("dataset/crypto_bot.csv", ';', 4);

    omnilearn::NetworkParam netp;

    netp.threads = 4;
    netp.batchSize = 50;
    netp.batchSizePercent = false;
    netp.learningRate = 0.01;
    netp.loss = omnilearn::Loss::BinaryCrossEntropy;
    netp.patience = 5 ;
    netp.improvement = 0.01;

    netp.momentum = 0.;
    netp.momentumScheduler = omnilearn::Scheduler::Exp;
    netp.momentumSchedulerDelay = 1;
    netp.momentumSchedulerValue = 0.005;
    netp.maxMomentum = 0.9;

    netp.scheduler = omnilearn::Scheduler::Exp;
    netp.schedulerValue = 0.005;
    //netp.schedulerDelay = 1;
    netp.scheduleBatchSize = true;
    netp.scheduleLearningRate = true;
    netp.maxBatchSize = 10;
    netp.maxBatchSizePercent = true;

    netp.classificationThreshold = 0.50;
    netp.validationRatio = 0.10;
    netp.testRatio = 0.25;
    netp.verbose = true;

    netp.preprocessInputs = {omnilearn::Preprocess::Standardize, omnilearn::Preprocess::Whiten, omnilearn::Preprocess::Reduce};
    netp.inputReductionThreshold = 0.95;
    netp.inputWhiteningType = omnilearn::WhiteningType::PCA;
    netp.optimizer = omnilearn::Optimizer::Adadelta;

    //netp.weightMode = omnilearn::Weight::Automatic;
    //netp.weightMode = omnilearn::Weight::Enabled;
    //netp.weights = (omnilearn::Vector(9) << 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25).finished();
    //netp.weights = (omnilearn::Vector(1) << 0.15).finished();

    netp.dropout = 0.5;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();

    lay.size = 512;
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Prelu;
    net.addLayer(lay);

    lay.size = 512;
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Prelu;
    net.addLayer(lay);

    // output layer
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Sigmoid;
    net.addLayer(lay);

    net.learn();
}



void cube()
{
    omnilearn::Data data = omnilearn::loadData("dataset/cube.csv", ';', 3);

    omnilearn::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 1;
    netp.learningRate = 0.001;
    netp.loss = omnilearn::Loss::BinaryCrossEntropy;
    netp.patience = 10;
    netp.improvement = 0.01;

    netp.scheduler = omnilearn::Scheduler::Plateau;
    netp.schedulerValue = 2;
    netp.schedulerDelay = 2;

    netp.classificationThreshold = 0.50;
    netp.validationRatio = 0.1;
    netp.testRatio = 0.1;
    netp.verbose = true;
    //netp.weightMode = omnilearn::Weight::Automatic;
    netp.weightMode = omnilearn::Weight::Enabled;
    netp.weights = (omnilearn::Vector(8) << 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33).finished();

    netp.optimizer = omnilearn::Optimizer::Adadelta;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    lay.size = 32;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Softplus;
    net.addLayer(lay);

    lay.size = 16;
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Softplus;
    net.addLayer(lay);

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Sigmoid;
    net.addLayer(lay);

    net.learn();
}


int main()
{
    //mnist();
    vesta();
    //iris();
    //testLoader();
    //generate();
    //cryptoBot();
    //cube();

    return 0;
}