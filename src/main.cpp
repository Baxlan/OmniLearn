
#include <fstream>
#include <map>
#include "brain/brain.hh"
#include "brain/json.hh"


using json = nlohmann::json;


enum class DataType {Iris, Vesta};


brain::Dataset extractData(DataType dataType)
{
    brain::Dataset data;

    if(dataType == DataType::Iris)
    {
        std::fstream dataFile("dataset/iris.json");
        json dataset = json::parse(dataFile)["list"];

        std::map<std::string, std::vector<double>> dataRes;
        dataRes["setosa"] = {1, 0, 0};
        dataRes["versicolor"] = {0, 1, 0};
        dataRes["virginica"] = {0, 0, 1};

        data = brain::Dataset(dataset.size());

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
            for(unsigned i = 0; i < 14; i++)
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
    brain::Dataset data(extractData(DataType::Vesta));

    std::vector<std::string> labels = {"Mo95","Tc99","Ru101","Rh103","Ag109","Cs133","Nd143","Sm147","Sm149","Sm150","Sm151","Sm152","Gd155","Eu153","U235","U238","Pu238","Pu239","Pu240","Pu241","Pu242","Am241","Am242"};
    //std::vector<std::string> labels = {"setosa", "versicolor", "virginica"};

    brain::NetworkParam netp;
    netp.threads = 1;
    netp.batchSize = 10;
    netp.learningRate = 0.001;
    netp.loss = brain::Loss::L2;
    netp.epoch = 200;
    netp.patience = 10;
    netp.decay = brain::LRDecay::exp;
    netp.LRDecayConstant = 0.3;
    netp.classValidity = 0.90;
    netp.validationRatio = 0.2;
    netp.testRatio = 0.2;
    netp.optimizer = brain::Optimizer::Rmsprop;
    netp.metric = brain::Metric::L1;

    brain::Network net(labels, netp);
    net.setData(data);

    brain::LayerParam lay;
    lay.size = 32;
    lay.maxNorm = 5;
    net.addLayer<brain::Dot, brain::Relu>(lay);
    lay.size = 23;
    net.addLayer<brain::Dot, brain::Linear>(lay);


    if(net.learn())
    {
        net.writeInfo("output.out");
    }

    return 0;
}