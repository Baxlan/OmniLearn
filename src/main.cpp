
#include "brain/brain.hh"


int main()
{
    brain::Data data = brain::loadData("dataset/mnist_train.csv", ',');

    brain::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 500;
    netp.learningRate = 0.0001;
    netp.loss = brain::Loss::CrossEntropy;
    netp.epoch = 500;
    netp.patience = 10;
    netp.decay = brain::decay::exp;
    netp.decayValue = 0.5;
    netp.classValidity = 0.70;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.10;
    netp.optimizer = brain::Optimizer::Rmsprop;
    netp.metric = brain::Metric::Accuracy;
    netp.preprocess = {brain::Preprocess::Standardize};
    netp.normalizeOutputs = false;

    brain::Network net(data, netp);

    brain::LayerParam lay;
    lay.maxNorm = 5;
    lay.size = 128;
    net.addLayer<brain::Dot, brain::Relu>(lay);
    lay.size = 32;
    net.addLayer<brain::Dot, brain::Relu>(lay);
    lay.size = 10;
    net.addLayer<brain::Dot, brain::Linear>(lay);


    if(net.learn())
    {
        net.writeInfo("output.out");
    }

    return 0;
}