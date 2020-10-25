// main.cpp

#include "omnilearn/Network.hh"
#include "omnilearn/metric.h"



void boston()
{
    omnilearn::Data data = omnilearn::loadData("dataset/boston.csv", ',', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 5;
    netp.learningRate = 0.005;
    netp.loss = omnilearn::Loss::L2;
    netp.patience = 5;
    netp.plateau = 1.00;
    netp.decay = omnilearn::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 2;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.15;
    netp.preprocessInputs = {omnilearn::Preprocess::Standardize};
    netp.preprocessOutputs = {omnilearn::Preprocess::Normalize};
    netp.verbose = true;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay;
    lay.size = 32;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Relu;
    net.addLayer(lay);

    lay.activation = omnilearn::Activation::Linear;
    net.addLayer(lay);

    net.learn();
}


void vesta()
{
    omnilearn::Data data = omnilearn::loadData("dataset/vesta.csv", ';', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 10;
    netp.learningRate = 0.001;
    netp.loss = omnilearn::Loss::L2;
    netp.patience = 5;
    netp.plateau = 0.99;
    netp.decay = omnilearn::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 1;
    netp.validationRatio = 0.20;
    netp.testRatio = 0.20;
    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Whiten};
    netp.preprocessOutputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Normalize};
    netp.verbose = true;
    netp.adaptiveLearningRate = true;
    netp.window = 0.999;
    netp.momentum = 0.9;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay;
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
    netp.threads = 4;
    netp.batchSize = 10;
    netp.learningRate = 0.01;
    netp.loss = omnilearn::Loss::CrossEntropy;
    netp.patience = 100;
    netp.plateau = 0.99;
    //netp.decay = omnilearn::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 2;
    netp.classValidity = 0.50;
    netp.validationRatio = 0.20;
    netp.testRatio = 0.20;
    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Reduce};
    netp.preprocessOutputs = {};
    netp.inputReductionThreshold = 0.99;
    netp.verbose = true;
    netp.momentum = 0;
    netp.adaptiveLearningRate = true;
    netp.automaticLearningRate = true;

    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);

    omnilearn::LayerParam lay;
    lay.size = 32;
    lay.lockBias = true;
    lay.lockWeights = true;

    lay.aggregation = omnilearn::Aggregation::Dot;
    lay.activation = omnilearn::Activation::Pelu;
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
    netp.batchSize = 100;
    netp.learningRate = 0.01;
    netp.loss = omnilearn::Loss::CrossEntropy;
    netp.epoch = 500;
    netp.patience = 5;
    netp.plateau = 0.99;
    netp.decay = omnilearn::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 1;
    netp.classValidity = 0.50;
    netp.validationRatio = 0.20;
    netp.testRatio = 0.0;
    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Reduce};
    netp.preprocessOutputs = {};
    netp.inputReductionThreshold = 0.99;
    netp.verbose = true;

    netp.adaptiveLearningRate = true;
    netp.momentum = 0.9;
    netp.window = 0.99;



    omnilearn::Network net;
    net.setParam(netp);
    net.setData(data);
    net.setTestData(testdata);

    omnilearn::LayerParam lay;
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
    std::pair<double, double> metric = omnilearn::classificationMetrics(data.outputs, genNet.process(data.inputs), 0.5);
    std::cout << metric.first << " " << metric.second << "\n";
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


int main()
{
    //mnist();
    //vesta();
    iris();
    //testLoader();
    //generate();

    return 0;
}