

#define BRAIN_ENABLE_SVD
#include "brain/brain.hh"


void vesta()
{
    brain::Data data = brain::loadData("dataset/vesta.csv", ';', 4);

    brain::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 10;
    netp.learningRate = 0.001;
    netp.loss = brain::Loss::L2;
    netp.patience = 10;
    netp.plateau = 0.99;
    netp.decay = brain::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 2;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.15;
    netp.optimizer = brain::Optimizer::Rmsprop;
    netp.preprocessInputs = {brain::Preprocess::Center, brain::Preprocess::Decorrelate, brain::Preprocess::Whiten};
    netp.preprocessOutputs = {brain::Preprocess::Center, brain::Preprocess::Decorrelate, brain::Preprocess::Normalize};

    brain::Network net(data, netp);

    brain::LayerParam lay;
    lay.maxNorm = 5;
    lay.size = 32;
    net.addLayer<brain::Dot, brain::Relu>(lay);
    net.addLayer<brain::Dot, brain::Linear>(lay);


    net.learn();
}


void mnist()
{
    brain::Data data = brain::loadData("dataset/mnist_train.csv", ',', 4);
    brain::Data testdata = brain::loadData("dataset/mnist_test.csv", ',', 4);

    brain::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 100;
    netp.learningRate = 0.0002;
    netp.loss = brain::Loss::CrossEntropy;
    netp.epoch = 500;
    netp.patience = 10;
    netp.plateau = 0.99;
    netp.decay = brain::Decay::Plateau;
    netp.decayValue = 2;
    netp.decayDelay = 2;
    netp.classValidity = 0.80;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.0;
    netp.optimizer = brain::Optimizer::Rmsprop;
    netp.preprocessInputs = {brain::Preprocess::Center, brain::Preprocess::Decorrelate, brain::Preprocess::Reduce};
    netp.preprocessOutputs = {};
    netp.inputReductionThreshold = 0.99;

    brain::Network net(data, netp);
    net.setTestData(testdata);

    brain::LayerParam lay;
    lay.maxNorm = 5;
    lay.size = 300;
    net.addLayer<brain::Dot, brain::Relu>(lay);
    net.addLayer<brain::Dot, brain::Linear>(lay);

    net.learn();
}

int main()
{
    mnist();
    //vesta();

    return 0;
}