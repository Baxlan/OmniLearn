//main.cpp

#include "burnet.hh"
#include "json.hh"


#include <iostream>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <fstream>
#include <iomanip>

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

    return 0;
}