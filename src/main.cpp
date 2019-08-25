
#include <fstream>
#include <map>
#include "burnet/burnet.hh"
#include "burnet/json.hh"


using json = nlohmann::json;


enum class DataType {Iris, Vesta};


burnet::Dataset extractData(DataType dataType)
{
    burnet::Dataset data;

    if(dataType == DataType::Iris)
    {
        std::fstream dataFile("dataset/iris.json");
        json dataset = json::parse(dataFile)["list"];

        std::map<std::string, std::vector<double>> dataRes;
        dataRes["setosa"] = {1, 0, 0};
        dataRes["versicolor"] = {0, 1, 0};
        dataRes["virginica"] = {0, 0, 1};

        data = burnet::Dataset(dataset.size());

        for(unsigned i = 0; i < dataset.size(); i++)
        {
            data[i] = {{dataset[i]["sepal_length"], dataset[i]["sepal_width"], dataset[i]["petal_length"], dataset[i]["petal_width"]}, dataRes[dataset[i]["species"]]};
        }
    }
    if(dataType == DataType::Vesta)
    {
        std::fstream dataFile("dataset/vesta.csv");
        std::vector<std::string> content;
        while(dataFile.good())
        {
            content.push_back("");
            std::getline(dataFile, content[content.size()-1]);
        }

        //remove title line
        content.erase(content.begin());

        for(std::string line : content)
        {
            line+=";";
            std::vector<double> inputs;
            std::vector<double> outputs;

            //time
            inputs.push_back(std::stod(line.substr(0, line.find(";"))));
            line.erase(0, line.find(";") + 1);

            for(unsigned i = 1; i < 24; i++)
            {
                outputs.push_back(std::stod(line.substr(0, line.find(";"))));
                line.erase(0, line.find(";") + 1);
            }
            for(unsigned i = 0; i < 17; i++)
            {
                inputs.push_back(std::stod(line.substr(0, line.find(";"))));
                line.erase(0, line.find(";") + 1);
            }
            data.push_back({inputs, outputs});
        }
    }
    return data;
}


int main()
{
    burnet::Dataset data(extractData(DataType::Vesta));

    burnet::nthreads = 4;

    burnet::NetworkParam netp;
    netp.batchSize = 1;
    netp.learningRate = 0.001;
    netp.dropout = 0.0;
    netp.dropconnect = 0.0;
    netp.loss = burnet::Loss::L2;
    netp.L2 = 0.;
    netp.maxEpoch = 200;
    netp.epochAfterOptimal = 20;
    netp.decay = burnet::LRDecay::inverse;
    netp.LRDecayConstant = 1.0;
    netp.margin = 10;
    netp.validationRatio = 0.2;
    netp.testRatio = 0.1;

    burnet::Network net(netp);
    net.setData(data);

    burnet::LayerParam lay;
    lay.size = 32;
    lay.maxNorm = 5;
    net.addLayer<burnet::Dot, burnet::Relu>(lay);
    lay.size = 23;
    net.addLayer<burnet::Dot, burnet::Sigmoid>(lay);


    if(net.learn())
    {
        net.writeInfo("output.txt");
    }

    return 0;
}