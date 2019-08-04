
#include <fstream>
#include <map>
#include "burnet/burnet.hh"
#include "burnet/json.hh"


using json = nlohmann::json;

int main()
{
    std::fstream dataFile("dataset/iris.json");
    json dataset = json::parse(dataFile)["list"];

    std::map<std::string, std::vector<double>> dataRes;
    dataRes["setosa"] = {1, 0, 0};
    dataRes["versicolor"] = {0, 1, 0};
    dataRes["virginica"] = {0, 0, 1};


    burnet::Dataset data(dataset.size());

    for(unsigned i = 0; i < dataset.size(); i++)
    {
        data[i] = {{dataset[i]["sepal_length"], dataset[i]["sepal_width"], dataset[i]["petal_length"], dataset[i]["petal_width"]}, dataRes[dataset[i]["species"]]};
    }

    burnet::decayParam::a = 0.05;

    burnet::NetworkParam netp;
    netp.batchSize = 1;
    netp.learningRate = 0.1;
    netp.dropout = 0;
    netp.dropconnect = 0.2;
    netp.loss = burnet::Loss::CrossEntropy;
    netp.L2 = 0.0001;
    netp.maxEpoch = 100000;
    netp.epochAfterOptimal = 700;
    //netp.decay = burnet::noDecay;

    burnet::Network net(data, netp);

    burnet::LayerParam lay;
    lay.size = 4;
    lay.maxNorm = 4;
    lay.k = 1;
    net.addLayer<burnet::Dot, burnet::Relu>(lay);
    lay.size = 3;
    net.addLayer<burnet::Dot, burnet::Linear>(lay);
    net.learn();


    return 0;
}