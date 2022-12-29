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
    netp.batchSize = 1;
    netp.maxBatchSizeRatio = 0.1;
    netp.scheduleBatchSize = true;
    netp.learningRate = 0.01;
    netp.scheduler = omnilearn::Scheduler::Plateau;
    netp.schedulerValue = 2;
    netp.schedulerDelay = 2;

    netp.momentum = 0;
    netp.maxMomentum = 0.90;
    netp.momentumScheduler = omnilearn::Scheduler::Exp;
    netp.momentumSchedulerValue = 0.1;

    netp.loss = omnilearn::Loss::L2;
    netp.patience = 10;
    netp.improvement = 0.01;

    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Whiten};
    netp.preprocessOutputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Normalize};

    netp.verbose = true;
    netp.optimizer = omnilearn::Optimizer::AMSGrad;
    netp.window = 0.99;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    lay.size = 32;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Relu;
    net.addLayer(lay);

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Linear;
    net.addLayer(lay);

    net.learn();
}


void iris()
{
    omnilearn::Data data = omnilearn::loadData("dataset/iris.csv", ',', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 3;
    netp.batchSize = 1;
    netp.learningRate = 0.01;
    netp.loss = omnilearn::Loss::CrossEntropy;
    netp.patience = 10;
    netp.improvement = 0.01;
    netp.scheduler = omnilearn::Scheduler::Plateau;
    netp.schedulerValue = 2;
    netp.schedulerDelay = 2;
    netp.classificationThreshold = 0.50;
    netp.validationRatio = 0.20;
    netp.testRatio = 0.20;
    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Reduce};
    netp.preprocessOutputs = {};
    netp.inputReductionThreshold = 0.99;
    netp.verbose = true;
    netp.momentum = 0.9;

    netp.optimizer = omnilearn::Optimizer::Adadelta;
    netp.weightMode = omnilearn::Weight::Automatic;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    lay.size = 32;
    //lay.lockBias = true;
    //lay.lockWeights = true;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Relu;
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
    netp.maxBatchSizeRatio = 0.1;
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

    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Reduce};
    netp.preprocessOutputs = {};
    netp.inputReductionThreshold = 0.99;

    netp.verbose = true;
    netp.optimizer = omnilearn::Optimizer::AMSGrad;
    netp.window = 0.99;



    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);
    net.setTestData(testdata);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    lay.maxNorm = 100;
    lay.size = 256;

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
    genNet.load("omnilearn_network", 4);
    omnilearn::Data data = omnilearn::loadData("dataset/mnist_test.csv", ',', 4);
    std::array<double, 4> metric = omnilearn::classificationMetrics(data.outputs, genNet.process(data.inputs), 0.5);
    std::cout << metric[0] << " " << metric[1] << " " << metric[2] << " " << metric[3] << "\n";
}


void generate()
{
    std::srand(static_cast<unsigned>(time(0)));
    omnilearn::Network genNet;
    genNet.load("omnilearn_network", 4);
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
    netp.batchSize = 0;
    netp.learningRate = 0.01;
    netp.loss = omnilearn::Loss::BinaryCrossEntropy;
    netp.patience = 20;
    netp.improvement = 0.01;

    netp.scheduler = omnilearn::Scheduler::Plateau;
    netp.schedulerValue = 1.5;
    netp.schedulerDelay = 1;
    netp.scheduleBatchSize = true;
    netp.scheduleLearningRate = true;

    netp.classificationThreshold = 0.50;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.15;
    netp.verbose = true;

    netp.preprocessInputs = {omnilearn::Preprocess::Standardize};
    netp.optimizer = omnilearn::Optimizer::Nadam;

    netp.weightMode = omnilearn::Weight::Automatic;
    //netp.weightMode = omnilearn::Weight::Enabled;
    //netp.weights = (omnilearn::Vector(8) << 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15).finished();
    //netp.weights = (omnilearn::Vector(1) << 0.15).finished();

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();

    lay.size = 64;
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Softplus;
    net.addLayer(lay);

    lay.size = 32;
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Sigmoid;
    net.addLayer(lay);

    net.learn();
}


void waterQuality()
{
    omnilearn::Data data = omnilearn::loadData("dataset/water-quality.csv", ';', 3);

    omnilearn::NetworkParam netp;
    netp.threads = 3;
    netp.batchSize = 1;
    netp.learningRate = 0.001;
    netp.loss = omnilearn::Loss::BinaryCrossEntropy;
    netp.patience = 15;
    netp.improvement = 0.01;
    netp.classificationThreshold = 0.50;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.15;
    netp.verbose = true;
    netp.weightMode = omnilearn::Weight::Automatic;

    netp.scheduleBatchSize = true;
    netp.scheduleLearningRate = true;
    netp.scheduler = omnilearn::Scheduler::Plateau;
    netp.schedulerValue = 1.5;
    netp.schedulerDelay = 4;

    netp.optimizer = omnilearn::Optimizer::AMSGrad;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay = omnilearn::Layer::generateNonLinearLayerParam();
    lay.size = 32;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Softplus;
    net.addLayer(lay);

    lay.size = 32;
    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Softplus;
    net.addLayer(lay);

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
    netp.validationRatio = 0.15;
    netp.testRatio = 0.15;
    netp.verbose = true;
    netp.weightMode = omnilearn::Weight::Automatic;

    netp.optimizer = omnilearn::Optimizer::Nadam;

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
    //vesta();
    //iris();
    //testLoader();
    //generate();
    cryptoBot();
    //waterQuality();
    //cube();

    return 0;
}