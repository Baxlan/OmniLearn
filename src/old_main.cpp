//main.cpp

#include "old/old.hh"
#include "old/json.hh"


#include <iostream>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <fstream>
#include <iomanip>

namespace learn = stb::learn;
using json = nlohmann::json;


int main()
{
    std::fstream settingFile("settings.json");
    json j = json::parse(settingFile);

    double L1Regularization = j["L1_regularization"];
    double L2Regularization = j["L2_regularization"];
    double learningRate = j["learning_rate"];
    double learningMargin = j["learning_margin"];
    double testingMargin = j["testing_margin"];
    unsigned nbEpoch = j["nb_epoch"];
    double tackOn = j["tackOn"];
    unsigned datasetSeed = j["dataset_seed"];
    double maxNorm = j["maxNorm"];
    // batchSize ?

    learn::NeuralFunctions func(learn::relu, learn::reluPrime);
    learn::NeuralFunctions funcOut(learn::sigmoid, learn::sigmoidPrime);
    learn::WeightInitializer init(0, 1, j["weights_seed"], 0);

    learn::NeuronProportion prop(func, init, 1);
    learn::NeuronProportion propOut(funcOut, init, 1);
    learn::LayerSettings settings(j["dropOut"], j["dropConnect"]);

    learn::Layer L1({prop}, 10, 4, settings);
    learn::Layer L2({prop}, 10, 10, settings);
    learn::Layer out({propOut}, 3, 10);

    std::fstream datasetFile("dataset/iris.json");
    json dataset = json::parse(datasetFile)["list"];

    // selecting validation test
    if(datasetSeed == 0)
        std::srand(static_cast<unsigned>(std::time(nullptr)));
    else
        std::srand(datasetSeed);

    std::vector<std::vector<double>> validationData;
    std::vector<std::vector<double>> validationResult;

    for(unsigned i=0; i<15; i++)
    {
        unsigned random = static_cast<unsigned>(std::rand()) % dataset.size();
        validationData.push_back({dataset[random]["sepal_length"], dataset[random]["sepal_width"], dataset[random]["petal_length"], dataset[random]["petal_width"]});

        if(dataset[random]["species"] == "setosa")
            validationResult.push_back({1, 0, 0});
        else if(dataset[random]["species"] == "versicolor")
            validationResult.push_back({0, 1, 0});
        else if(dataset[random]["species"] == "virginica")
            validationResult.push_back({0, 0, 1});

        dataset.erase(dataset.begin() + static_cast<int>(i));
    }

    std::vector<double> epochLoss(nbEpoch, 0);
    std::vector<double> epochCorrectedLoss(nbEpoch, 0);
    for(unsigned epoch=0; epoch<nbEpoch; epoch++)
    {
        std::cout << epoch << "\n";
        for(unsigned dataIndex = 0; dataIndex < dataset.size(); dataIndex++)
        {
            std::vector<double> data({dataset[dataIndex]["sepal_length"], dataset[dataIndex]["sepal_width"], dataset[dataIndex]["petal_length"], dataset[dataIndex]["petal_width"]});

            std::vector<double> result;
            if(dataset[dataIndex]["species"] == "setosa")
                result = {1, 0, 0};
            else if(dataset[dataIndex]["species"] == "versicolor")
                result = {0, 1, 0};
            else if(dataset[dataIndex]["species"] == "virginica")
                result = {0, 0, 1};

            std::vector<double> score = out.processToLearn(L2.processToLearn(L1.processToLearn(data)));
            std::vector<double> loss = learn::cost(result, score, learningMargin);
            double totalLoss = learn::absoluteSum(loss);
            double correctedLoss = totalLoss + L1Regularization * (out.getAbsoluteWeightsSum() + L1.getAbsoluteWeightsSum() + L2.getAbsoluteWeightsSum())
            + 0.5 * L2Regularization * (out.getSquaredWeightsSum() + L1.getSquaredWeightsSum() + L2.getSquaredWeightsSum());


            //backpropagate deltas
            out.setDelta(loss);
            L2.setDelta(out.getNextDelta());
            L1.setDelta(L2.getNextDelta());

            //then update the weights
            out.updateWeights(learningRate, L1Regularization, L2Regularization, tackOn, maxNorm);
            L1.updateWeights(learningRate, L1Regularization, L2Regularization, tackOn, maxNorm);
            L2.updateWeights(learningRate, L1Regularization, L2Regularization, tackOn, maxNorm);

            epochLoss[epoch] += totalLoss;
            epochCorrectedLoss[epoch] += correctedLoss;
        }
    }


    double validated = 0;
    for(unsigned i=0; i<validationData.size(); i++)
    {
        std::vector<double> score = out.process(L2.process(L1.process(validationData[i])));
        std::vector<double> loss = learn::cost(validationResult[i], score, testingMargin);
        double totalLoss = learn::absoluteSum(loss);

        double correctedLoss = totalLoss + L1Regularization * (out.getAbsoluteWeightsSum() + L1.getAbsoluteWeightsSum() + L2.getAbsoluteWeightsSum())
        + L2Regularization * (out.getSquaredWeightsSum() + L1.getSquaredWeightsSum() + L2.getSquaredWeightsSum());

        std::cout << "test " << std::setfill(' ') << std::setw(4)<< i << " : "
        << std::setfill(' ') << std::setw(15) << score[0]
        << std::setfill(' ') << std::setw(15) << score[1]
        << std::setfill(' ') << std::setw(15) << score[2]
        << "     true result {" << validationResult[i][0] << " " << validationResult[i][1] << " " << validationResult[i][2] << "}"
        << "     loss 1 : " << std::setfill(' ') << std::setw(10) << loss[0]
        << "     loss 2 : " << std::setfill(' ') << std::setw(10) << loss[1]
        << "     loss 3 : " << std::setfill(' ') << std::setw(10) << loss[2]
        << "     total loss : " << std::setfill(' ') << std::setw(10) << totalLoss
        << "     corrected loss : " << std::setfill(' ') << std::setw(10) << correctedLoss << "\n";

        if(totalLoss < 3*testingMargin)
            validated++;
    }

    std::cout << validated*100/15 << " %\n";



    std::ofstream output("output.txt");
    for(unsigned i=0; i<epochLoss.size(); i++)
    {
        output << epochLoss[i] << ",";
    }
    output << "\n";
    for(unsigned i=0; i<epochCorrectedLoss.size(); i++)
    {
        output << epochCorrectedLoss[i] << ",";
    }
    return 0;
}