
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

    burnet::Network net(data);
    net.shuffleData();
    net.addLayer<burnet::Dot, burnet::Relu>();
    net.addLayer<burnet::Dot, burnet::Relu>();
    net.initLayers();
    net.learn();

    return 0;
}