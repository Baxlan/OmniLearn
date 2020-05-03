// NetworkIO.cpp

#include "omnilearn/NetworkIO.hh"
#include "omnilearn/Network.hh"



omnilearn::NetworkIO::NetworkIO(std::string const& path):
_path(path),
_listing(path + ".listing", std::fstream::in | std::fstream::out |std::fstream::app),
_save(path + ".save", std::fstream::in | std::fstream::out | std::fstream::app),
_test(path + ".test", std::fstream::in | std::fstream::out | std::fstream::app)
{
  if(!_listing)
    throw Exception("Cannot access/create file " + path + ".listing");
  if(!_save)
    throw Exception("Cannot access/create file " + path + ".listing");
  if(!_test)
    throw Exception("Cannot access/create file " + path + ".listing");
}


void omnilearn::NetworkIO::list(std::string const& line)
{
  _listing << line << "\n";
}


void omnilearn::NetworkIO::saveNet(Network const& net)
{
  json jObj;

  jObj["parameters"] = {};
  jObj["preprocess"]["input"] = {};
  jObj["preprocess"]["output"] = {};
  jObj["coefs"] = {};

  saveParameters(net, jObj["parameters"]);
  saveInputPreprocess(net, jObj["preprocess"]["input"]);
  saveOutputPreprocess(net, jObj["preprocess"]["output"]);
  saveCoefs(net, jObj["coefs"]);

  _save << jObj;
}


void omnilearn::NetworkIO::saveTest(Network const& net)
{
  Matrix testRes(net.process(net._testRawInputs));
  for(size_t i = 0; i < net._outputLabels.size(); i++)
  {
    _test << "label: " << net._outputLabels[i] << "\n" ;
    for(eigen_size_t j = 0; j < net._testRawOutputs.rows(); j++)
      _test << net._testRawOutputs(j,i) << ",";
    _test << "\n";
    for(eigen_size_t j = 0; j < testRes.rows(); j++)
      _test << testRes(j,i) << ",";
    _test << "\n";
  }
}


void omnilearn::NetworkIO::load(Network& net) const
{
  json jObj = json::parse(std::ifstream(_path + ".save"));

  loadParameters(net, jObj.at("parameters"));
  loadInputPreprocess(net, jObj.at("preprocess").at("input"));
  loadOutputPreprocess(net, jObj.at("preprocess").at("output"));
  loadCoefs(net, jObj.at("coefs"));
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== PRIVATE PART ============================================================
//=============================================================================
//=============================================================================
//=============================================================================



void omnilearn::NetworkIO::saveParameters(Network const& net, json& jObj)
{
  if(net._param.loss == Loss::BinaryCrossEntropy)
    jObj["loss"] = "binary cross entropy";
  else if(net._param.loss == Loss::CrossEntropy)
    jObj["loss"] = "cross entropy";
  else if(net._param.loss == Loss::L1)
    jObj["loss"] = "mae";
  else if(net._param.loss == Loss::L2)
    jObj["loss"] = "mse";

  jObj["input labels"] = net._inputLabels;
  jObj["output labels"] = net._outputLabels;
  jObj["train losses"] = net._trainLosses;
  jObj["validation losses"] = net._validLosses;
  jObj["test first metrics"] = net._testMetric;
  jObj["test second metrics"] = net._testSecondMetric;
  if(jObj["loss"] == "binary cross entropy" || jObj["loss"] == "cross entropy")
  {
    jObj["classification threshold"] = net._param.classValidity;
  }
  jObj["optimal epoch"] = net._optimalEpoch;
}


void omnilearn::NetworkIO::saveInputPreprocess(Network const& net, json& jObj)
{
  jObj["preprocess"] = json::array();
  for(size_t i = 0; i < net._param.preprocessInputs.size(); i++)
  {
    if(net._param.preprocessInputs[i] == Preprocess::Center)
    {
      jObj["preprocess"][i] = "center";
      jObj["center"] = net._inputCenter;
    }
    else if(net._param.preprocessInputs[i] == Preprocess::Normalize)
    {
      jObj["preprocess"][i] = "normalize";

      std::vector<double> vec(0);
      // transform vector of pairs into vector of first element of each pair
      std::transform(net._inputNormalization.begin(), net._inputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["normalization"][0] = vec;

      vec = std::vector<double>(0);
      // transform vector of pairs into vector of second element of each pair
      std::transform(net._inputNormalization.begin(), net._inputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["normalization"][1] = vec;
    }
    else if(net._param.preprocessInputs[i] == Preprocess::Standardize)
    {
      jObj["preprocess"][i] = "standardize";

      std::vector<double> vec(0);
      // transform vector of pairs into vector of first element of each pair
      std::transform(net._inputStandartization.begin(), net._inputStandartization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["standardization"][0] = vec;

      vec = std::vector<double>(0);
      // transform vector of pairs into vector of second element of each pair
      std::transform(net._inputStandartization.begin(), net._inputStandartization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["standardization"][1] = vec;
    }
    else if(net._param.preprocessInputs[i] == Preprocess::Decorrelate)
    {
      jObj["preprocess"][i] = "decorrelate";
      jObj["eigenvalues"] = net._inputDecorrelation.second;

      Matrix vectors = net._inputDecorrelation.first.transpose();
      for(eigen_size_t j = 0; j < vectors.rows(); j++)
      {
        jObj["eigenvectors"][j] = Vector(vectors.row(j));
      }
    }
    else if(net._param.preprocessInputs[i] == Preprocess::Whiten)
    {
      jObj["preprocess"][i] = "whiten";
      jObj["whitening bias"] = net._param.inputWhiteningBias;
    }
    else if(net._param.preprocessInputs[i] == Preprocess::Reduce)
    {
      jObj["preprocess"][i] = "reduce";
      jObj["reduction threshold"] = net._param.inputReductionThreshold;
    }
  }
}


void omnilearn::NetworkIO::saveOutputPreprocess(Network const& net, json& jObj)
{
  for(size_t i = 0; i < net._param.preprocessOutputs.size(); i++)
  {
    if(net._param.preprocessOutputs[i] == Preprocess::Center)
    {
      jObj["preprocess"][i] = "center";
      jObj["center"] = net._outputCenter;
    }
    else if(net._param.preprocessOutputs[i] == Preprocess::Normalize)
    {
      jObj["preprocess"][i] = "normalize";

      std::vector<double> vec(0);
      // transform vector of pairs into vector of first element of each pair
      std::transform(net._outputNormalization.begin(), net._outputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["normalization"][0] = vec;

      vec = std::vector<double>(0);
      // transform vector of pairs into vector of second element of each pair
      std::transform(net._outputNormalization.begin(), net._outputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["normalization"][1] = vec;
    }
    else if(net._param.preprocessOutputs[i] == Preprocess::Decorrelate)
    {
      jObj["preprocess"][i] = "decorrelate";
      jObj["eigenvalues"] = net._outputDecorrelation.second;

      Matrix vectors = net._outputDecorrelation.first.transpose();
      for(eigen_size_t k = 0; k < vectors.rows(); k++)
      {
        jObj["eigenvectors"][k] = Vector(vectors.row(k));
      }
    }
    else if(net._param.preprocessOutputs[i] == Preprocess::Reduce)
    {
      jObj["preprocess"][i] = "reduce";
      jObj["reduction threshold"] = net._param.outputReductionThreshold;
    }
  }
}


void omnilearn::NetworkIO::saveCoefs(Network const& net, json& jObj)
{
  for(size_t i = 0; i < net._layers.size(); i++)
  {
    jObj[i] = net._layers[i];
  }
}


void omnilearn::NetworkIO::loadParameters(Network& net, json const& jObj) const
{
  if(jObj.at("loss") == "binary cross entropy")
    net._param.loss = Loss::BinaryCrossEntropy;
  else if(jObj.at("loss") == "cross entropy")
    net._param.loss = Loss::CrossEntropy;
  else if(jObj.at("loss") == "mae")
    net._param.loss = Loss::L1;
  else if(jObj.at("loss") == "mse")
    net._param.loss = Loss::L2;

  jObj.at("input labels").get_to(net._inputLabels);
  jObj.at("output labels").get_to(net._outputLabels);
  if(net._param.loss == Loss::BinaryCrossEntropy || net._param.loss == Loss::CrossEntropy)
  {
    net._param.classValidity = jObj.at("classification threshold");
  }
  net._optimalEpoch = jObj.at("optimal epoch");
}


void omnilearn::NetworkIO::loadInputPreprocess(Network& net, json const& jObj) const
{
  net._param.preprocessInputs.resize(jObj.at("preprocess").size());
  for(size_t i = 0; i < jObj.at("preprocess").size(); i++)
  {
    if(jObj.at("preprocess").at(i) == "center")
    {
      net._param.preprocessInputs[i] = Preprocess::Center;
      net._inputCenter = stdToEigenVector(jObj.at("center"));
    }
    else if(jObj.at("preprocess").at(i) == "normalize")
    {
      net._param.preprocessInputs[i] = Preprocess::Normalize;
      std::vector<double> vec0 = jObj.at("normalization").at(0);
      std::vector<double> vec1 = jObj.at("normalization").at(1);
      std::transform(vec0.begin(), vec0.end(), vec1.begin(), std::back_inserter(net._inputNormalization), [](double a, double b)->std::pair<double, double>{return std::make_pair(a, b);});
    }
    else if(jObj.at("preprocess").at(i) == "standardize")
    {
      net._param.preprocessInputs[i] = Preprocess::Normalize;
      std::vector<double> vec0 = jObj.at("standardize").at(0);
      std::vector<double> vec1 = jObj.at("standardize").at(1);
      std::transform(vec0.begin(), vec0.end(), vec1.begin(), std::back_inserter(net._inputStandartization), [](double a, double b)->std::pair<double, double>{return std::make_pair(a, b);});
    }
    else if(jObj.at("preprocess").at(i) == "decorrelate")
    {
      net._param.preprocessInputs[i] = Preprocess::Decorrelate;
      net._inputDecorrelation.second = stdToEigenVector(jObj.at("eigenvalues"));
      net._inputDecorrelation.first = Matrix(jObj.at("eigenvectors").size(), jObj.at("eigenvectors").at(0).size());
      for(eigen_size_t j = 0; j < net._inputDecorrelation.first.rows(); j++)
      {
        net._inputDecorrelation.first.row(j) = stdToEigenVector(jObj.at("eigenvectors").at(j));
      }
    }
    else if(jObj.at("preprocess").at(i) == "whiten")
    {
      net._param.preprocessInputs[i] = Preprocess::Whiten;
      net._param.inputWhiteningBias = jObj.at("whitening bias");
    }
    else if(jObj.at("preprocess").at(i) == "reduce")
    {
      net._param.preprocessInputs[i] = Preprocess::Reduce;
      net._param.inputReductionThreshold = jObj.at("reduction threshold");
    }
  }
}


void omnilearn::NetworkIO::loadOutputPreprocess(Network& net, json const& jObj) const
{
  net._param.preprocessOutputs.resize(jObj.at("preprocess").size());
  for(size_t i = 0; i < jObj.at("preprocess").size(); i++)
  {
    if(jObj.at("preprocess").at(i) == "center")
    {
      net._param.preprocessOutputs[i] = Preprocess::Center;
      net._outputCenter = stdToEigenVector(jObj.at("center"));
    }
    else if(jObj.at("preprocess").at(i) == "normalize")
    {
      net._param.preprocessOutputs[i] = Preprocess::Normalize;
      std::vector<double> vec0 = jObj.at("normalization").at(0);
      std::vector<double> vec1 = jObj.at("normalization").at(1);
      std::transform(vec0.begin(), vec0.end(), vec1.begin(), std::back_inserter(net._outputNormalization), [](double a, double b)->std::pair<double, double>{return std::make_pair(a, b);});
    }
    else if(jObj.at("preprocess").at(i) == "decorrelate")
    {
      net._param.preprocessOutputs[i] = Preprocess::Decorrelate;
      net._outputDecorrelation.second = stdToEigenVector(jObj.at("eigenvalues"));
      net._outputDecorrelation.first = Matrix(jObj.at("eigenvectors").size(), jObj.at("eigenvectors").at(0).size());
      for(eigen_size_t j = 0; j < net._outputDecorrelation.first.rows(); j++)
      {
        net._outputDecorrelation.first.row(j) = stdToEigenVector(jObj.at("eigenvectors").at(j));
      }
    }
    else if(jObj.at("preprocess").at(i) == "reduce")
    {
      net._param.preprocessOutputs[i] = Preprocess::Reduce;
      net._param.outputReductionThreshold = jObj.at("reduction threshold");
    }
  }
}


void omnilearn::NetworkIO::loadCoefs(Network& net, json const& jObj) const
{
  net._layers.resize(jObj.at("coefs").size());
  for(size_t i = 0; i < net._layers.size(); i++)
  {
    net._layers[i] = jObj.at("coefs").at(i);
  }
}