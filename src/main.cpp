
#include "omnilearn/Network.hh"


void vesta()
{
    omnilearn::Data data = omnilearn::loadData("dataset/vesta.csv", ';', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 10;
    netp.learningRate = 0.001;
    netp.loss = omnilearn::Loss::L2;
    netp.patience = 10;
    netp.plateau = 0.99;
    netp.decay = omnilearn::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 2;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.15;
    netp.optimizer = omnilearn::Optimizer::Rmsprop;
    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Whiten};
    netp.postprocessOutputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Normalize};

    omnilearn::Network net(data, netp);

    omnilearn::LayerParam lay;
    lay.maxNorm = 5;
    lay.size = 32;
    net.addLayer(lay, omnilearn::Aggregation::Dot, omnilearn::Activation::Relu);
    net.addLayer(lay, omnilearn::Aggregation::Dot, omnilearn::Activation::Linear);


    net.learn();
}


void mnist()
{
    omnilearn::Data data = omnilearn::loadData("dataset/mnist_train.csv", ',', 4);
    omnilearn::Data testdata = omnilearn::loadData("dataset/mnist_test.csv", ',', 4);

    omnilearn::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 100;
    netp.learningRate = 0.0002;
    netp.loss = omnilearn::Loss::CrossEntropy;
    netp.epoch = 500;
    netp.patience = 10;
    netp.plateau = 0.99;
    netp.decay = omnilearn::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 2;
    netp.classValidity = 0.80;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.0;
    netp.optimizer = omnilearn::Optimizer::Rmsprop;
    netp.preprocessInputs = {omnilearn::Preprocess::Center, omnilearn::Preprocess::Decorrelate, omnilearn::Preprocess::Reduce};
    netp.postprocessOutputs = {};
    netp.inputReductionThreshold = 0.99;

    omnilearn::Network net(data, netp);
    net.setTestData(testdata);

    omnilearn::LayerParam lay;
    lay.maxNorm = 5;
    lay.size = 300;
    net.addLayer(lay, omnilearn::Aggregation::Dot, omnilearn::Activation::Relu);
    net.addLayer(lay, omnilearn::Aggregation::Dot, omnilearn::Activation::Linear);

    net.learn();
}


void testLoader()
{
    omnilearn::Network genNet("omnilearn_network", 4);
    omnilearn::Data data = omnilearn::loadData("dataset/mnist_test.csv", ',', 4);
    std::pair<double, double> metric = omnilearn::classificationMetrics(data.outputs, genNet.process(data.inputs), 0.8);
    std::cout << metric.first << " " << metric.second << "\n";
}


void generate()
{
    omnilearn::Network genNet("omnilearn_network", 4);
    omnilearn::NetworkParam param;
    param.epoch = 1000;
    param.learningRate = 0.1;
    omnilearn::Vector target = (omnilearn::Vector(10) << 1,0,0,0,0,0,0,0,0,0).finished();
    omnilearn::Vector input = omnilearn::Vector::Random(28*28);

    for(size_t i = 0; i < input.size(); i++)
    {
        input[i] += 1;
        input[i] *= 127.5;
    }

    omnilearn::Vector res = genNet.generate(param, target, input);

    std::cout << res << "\n\n";
    std::cout << genNet.process(res) << "\n";
}


int main()
{
    //mnist();
    //vesta();
    //testLoader();
    generate();

    return 0;
}