
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

    burnet::NetworkParam netp;
    burnet::Network net(data, netp);

    burnet::LayerParam lay;
    lay.maxNorm = 1000;
    net.addLayer<burnet::Dot, burnet::Relu>(lay);
    lay.size = 3;
    net.addLayer<burnet::Dot, burnet::Sigmoid>(lay);
    net.learn();

    return 0;
}