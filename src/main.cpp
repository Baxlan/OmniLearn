
#include "brain/brain.hh"


int main()
{
    brain::Data data = brain::loadData("dataset/vesta.csv", ';');
    //brain::Data testdata = brain::loadData("dataset/mnist_test.csv", ',');

    brain::NetworkParam netp;
    netp.threads = 4;
    netp.batchSize = 10;
    netp.learningRate = 0.001;
    netp.loss = brain::Loss::L2;
    netp.epoch = 500;
    netp.patience = 5;
    netp.plateau = 0.99;
    netp.decay = brain::Decay::Inverse;
    netp.decayValue = 0.2;
    netp.decayDelay = 2;
    netp.classValidity = 0.80;
    netp.validationRatio = 0.15;
    netp.testRatio = 0.15;
    netp.optimizer = brain::Optimizer::Rmsprop;
    netp.preprocess = {brain::Preprocess::Standardize};
    netp.normalizeOutputs = true;

    brain::Network net(data, netp);
    //net.setTestData(testdata);

    brain::LayerParam lay;
    lay.maxNorm = 5;
    lay.size = 16;
    net.addLayer<brain::Dot, brain::Relu>(lay);
    lay.size = 23;
    net.addLayer<brain::Dot, brain::Linear>(lay);


    if(net.learn())
    {
        net.writeInfo("output.out");
    }

    return 0;
}