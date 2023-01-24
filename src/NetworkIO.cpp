// NetworkIO.cpp

#include "omnilearn/Exception.hh"
#include "omnilearn/NetworkIO.hh"
#include "omnilearn/Network.hh"


omnilearn::NetworkIO& omnilearn::operator<<(NetworkIO& io, std::string const& text)
{
  io._listing << text;
  if(io._verbose)
    std::cout << text;
  if(text.find_first_of('\n') != std::string::npos)
    io._listing.flush();
  return io;
}

omnilearn::NetworkIO& omnilearn::operator<<(NetworkIO& io, std::_Setw const& setw)
{
  io._listing << setw << std::left;
  if(io._verbose)
    std::cout << setw << std::left;
  return io;
}

omnilearn::NetworkIO& omnilearn::operator<<(NetworkIO& io, std::_Setprecision const& setp)
{
  io._listing << setp << std::left;
  if(io._verbose)
    std::cout << setp << std::left;
  return io;
}

omnilearn::NetworkIO::NetworkIO(fs::path const& path, bool verbose):
_path(path),
_verbose(verbose)
{
  //check if name is already taken. If yes, add a counter to the name
  if(fs::exists(path.string() + ".save") || fs::exists(path.string() + ".listing"))
  {
    size_t count = 2;
    while(true)
    {
      if(fs::exists(path.string() + std::to_string(count) + ".save") || fs::exists(path.string() + std::to_string(count) + ".listing"))
        count++;
      else
      {
        _path = path.string() + std::to_string(count);
        break;
      }
    }
  }

  //create/open files to be sure they are available
  _listing = std::ofstream(_path.string() + ".listing");
  std::ofstream save(_path.string() + ".save");

  if(!_listing)
    throw Exception("Cannot access/create file " + _path.string() + ".listing");
  if(!save)
    throw Exception("Cannot access/create file " + _path.string() + ".save");
}


void omnilearn::NetworkIO::save(Network const& net) const
{
  json jObj = {};

  saveParameters(net, jObj["parameters"]);
  saveInputPreprocess(net, jObj["preprocess"]["input"]);
  saveOutputPreprocess(net, jObj["preprocess"]["output"]);
  saveCoefs(net, jObj["coefs"]);
  saveTest(net, jObj["test"]);

  std::ofstream save(_path.string() + ".save");
  if(!save)
    throw Exception("Cannot access/create file " + _path.string() + ".save");

  save << jObj;
}


void omnilearn::NetworkIO::load(Network& net, fs::path const& path)
{
  json jObj = json::parse(std::ifstream(path.string() + ".save"));

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



void omnilearn::NetworkIO::saveParameters(Network const& net, json& jObj) const
{
  if(net._param.loss == Loss::BinaryCrossEntropy)
    jObj["loss"] = "binary cross entropy";
  else if(net._param.loss == Loss::CrossEntropy)
    jObj["loss"] = "cross entropy";
  else if(net._param.loss == Loss::L1)
    jObj["loss"] = "L1";
  else if(net._param.loss == Loss::L2)
    jObj["loss"] = "L2";

  if(net._param.inputWhiteningType == WhiteningType::ZCA)
    jObj["input whitening type"] = "ZCA";
  else
    jObj["input whitening type"] = "PCA";

  if(net._param.outputWhiteningType == WhiteningType::ZCA)
    jObj["output whitening type"] = "ZCA";
  else
    jObj["output whitening type"] = "PCA";

  jObj["input labels"] = net._inputLabels;
  jObj["output labels"] = net._outputLabels;
  jObj["train losses"] = net._trainLosses;
  jObj["validation losses"] = net._validLosses;
  jObj["test first metrics"] = net._testMetric;
  jObj["test second metrics"] = net._testSecondMetric;
  jObj["test third metrics"] = net._testThirdMetric;
  jObj["test fourth metrics"] = net._testFourthMetric;
  if(jObj["loss"] == "binary cross entropy" || jObj["loss"] == "cross entropy")
  {
    jObj["classification threshold"] = net._param.classificationThreshold;
  }
  jObj["optimal epoch"] = net._optimalEpoch;
}


void omnilearn::NetworkIO::saveInputPreprocess(Network const& net, json& jObj) const
{
  jObj["preprocess"] = json::array();
  for(size_t i = 0; i < net._param.preprocessInputs.size(); i++)
  {
    if(net._param.preprocessInputs[i] == Preprocess::Normalize)
    {
      jObj["preprocess"][i] = "normalize";

      std::vector<double> vec(0);
      // transform vector of pairs into vector of first element of each pair
      std::transform(net._inputNormalization.begin(), net._inputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["normalization"][0] = vec;

      vec = std::vector<double>(0);
      // transform vector of pairs into vector of second element of each pair
      std::transform(net._inputNormalization.begin(), net._inputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<1>));
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
      std::transform(net._inputStandartization.begin(), net._inputStandartization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<1>));
      jObj["standardization"][1] = vec;
    }
    else if(net._param.preprocessInputs[i] == Preprocess::Whiten)
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


void omnilearn::NetworkIO::saveOutputPreprocess(Network const& net, json& jObj) const
{
  jObj["preprocess"] = json::array();
  for(size_t i = 0; i < net._param.preprocessOutputs.size(); i++)
  {
    if(net._param.preprocessOutputs[i] == Preprocess::Normalize)
    {
      jObj["preprocess"][i] = "normalize";

      std::vector<double> vec(0);
      // transform vector of pairs into vector of first element of each pair
      std::transform(net._outputNormalization.begin(), net._outputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["normalization"][0] = vec;

      vec = std::vector<double>(0);
      // transform vector of pairs into vector of second element of each pair
      std::transform(net._outputNormalization.begin(), net._outputNormalization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<1>));
      jObj["normalization"][1] = vec;
    }
    else if(net._param.preprocessOutputs[i] == Preprocess::Whiten)
    {
      jObj["preprocess"][i] = "decorrelate";
      jObj["eigenvalues"] = net._outputDecorrelation.second;

      Matrix vectors = net._outputDecorrelation.first.transpose();
      for(eigen_size_t j = 0; j < vectors.rows(); j++)
      {
        jObj["eigenvectors"][j] = Vector(vectors.row(j));
      }
    }
    else if(net._param.preprocessOutputs[i] == Preprocess::Reduce)
    {
      jObj["preprocess"][i] = "reduce";
      jObj["reduction threshold"] = net._param.outputReductionThreshold;
    }
    else if(net._param.preprocessOutputs[i] == Preprocess::Standardize)
    {
      jObj["preprocess"][i] = "standardize";

      std::vector<double> vec(0);
      // transform vector of pairs into vector of first element of each pair
      std::transform(net._outputStandartization.begin(), net._outputStandartization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<0>));
      jObj["standardization"][0] = vec;

      vec = std::vector<double>(0);
      // transform vector of pairs into vector of second element of each pair
      std::transform(net._outputStandartization.begin(), net._outputStandartization.end(), std::back_inserter(vec), static_cast<const double& (*)(const std::pair<double, double>&)>(std::get<1>));
      jObj["standardization"][1] = vec;
    }
    else if(net._param.preprocessOutputs[i] == Preprocess::Whiten)
    {
      jObj["preprocess"][i] = "whiten";
      jObj["whitening bias"] = net._param.outputWhiteningBias;
    }
  }
}


void omnilearn::NetworkIO::saveCoefs(Network const& net, json& jObj) const
{
  for(size_t i = 0; i < net._layers.size(); i++)
  {
    jObj[i] = net._layers[i];
  }
}


void omnilearn::NetworkIO::loadParameters(Network& net, json const& jObj)
{
  if(jObj.at("loss") == "binary cross entropy")
    net._param.loss = Loss::BinaryCrossEntropy;
  else if(jObj.at("loss") == "cross entropy")
    net._param.loss = Loss::CrossEntropy;
  else if(jObj.at("loss") == "L1")
    net._param.loss = Loss::L1;
  else if(jObj.at("loss") == "L2")
    net._param.loss = Loss::L2;

  if(jObj.at("input whitening type") == "ZCA")
    net._param.inputWhiteningType = WhiteningType::ZCA;
  else
    net._param.inputWhiteningType = WhiteningType::PCA;

  if(jObj.at("output whitening type") == "ZCA")
    net._param.outputWhiteningType = WhiteningType::ZCA;
  else
    net._param.outputWhiteningType = WhiteningType::PCA;

  jObj.at("input labels").get_to(net._inputLabels);
  jObj.at("output labels").get_to(net._outputLabels);
  if(net._param.loss == Loss::BinaryCrossEntropy || net._param.loss == Loss::CrossEntropy)
  {
    net._param.classificationThreshold = jObj.at("classification threshold");
  }
  net._optimalEpoch = jObj.at("optimal epoch");
}


void omnilearn::NetworkIO::loadInputPreprocess(Network& net, json const& jObj)
{
  net._param.preprocessInputs.resize(jObj.at("preprocess").size());
  for(size_t i = 0; i < jObj.at("preprocess").size(); i++)
  {
    if(jObj.at("preprocess").at(i) == "normalize")
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
      net._param.preprocessInputs[i] = Preprocess::Whiten;
      net._inputDecorrelation.second = stdToEigenVector(jObj.at("eigenvalues"));
      net._inputDecorrelation.first = Matrix(jObj.at("eigenvectors").size(), jObj.at("eigenvectors").at(0).size());
      for(eigen_size_t j = 0; j < net._inputDecorrelation.first.rows(); j++)
      {
        net._inputDecorrelation.first.row(j) = stdToEigenVector(jObj.at("eigenvectors").at(j));
      }
      net._inputDecorrelation.first.transposeInPlace();
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


void omnilearn::NetworkIO::loadOutputPreprocess(Network& net, json const& jObj)
{
  net._param.preprocessOutputs.resize(jObj.at("preprocess").size());
  for(size_t i = 0; i < jObj.at("preprocess").size(); i++)
  {
    if(jObj.at("preprocess").at(i) == "normalize")
    {
      net._param.preprocessOutputs[i] = Preprocess::Normalize;
      std::vector<double> vec0 = jObj.at("normalization").at(0);
      std::vector<double> vec1 = jObj.at("normalization").at(1);
      std::transform(vec0.begin(), vec0.end(), vec1.begin(), std::back_inserter(net._outputNormalization), [](double a, double b)->std::pair<double, double>{return std::make_pair(a, b);});
    }
    else if(jObj.at("preprocess").at(i) == "decorrelate")
    {
      net._param.preprocessOutputs[i] = Preprocess::Whiten;
      net._outputDecorrelation.second = stdToEigenVector(jObj.at("eigenvalues"));
      net._outputDecorrelation.first = Matrix(jObj.at("eigenvectors").size(), jObj.at("eigenvectors").at(0).size());
      for(eigen_size_t j = 0; j < net._outputDecorrelation.first.rows(); j++)
      {
        net._outputDecorrelation.first.row(j) = stdToEigenVector(jObj.at("eigenvectors").at(j));
      }
      net._outputDecorrelation.first.transposeInPlace();
    }
    else if(jObj.at("preprocess").at(i) == "reduce")
    {
      net._param.preprocessOutputs[i] = Preprocess::Reduce;
      net._param.outputReductionThreshold = jObj.at("reduction threshold");
    }
    else if(jObj.at("preprocess").at(i) == "standardize")
    {
      net._param.preprocessOutputs[i] = Preprocess::Normalize;
      std::vector<double> vec0 = jObj.at("standardize").at(0);
      std::vector<double> vec1 = jObj.at("standardize").at(1);
      std::transform(vec0.begin(), vec0.end(), vec1.begin(), std::back_inserter(net._outputStandartization), [](double a, double b)->std::pair<double, double>{return std::make_pair(a, b);});
    }
    else if(jObj.at("preprocess").at(i) == "whiten")
    {
      net._param.preprocessOutputs[i] = Preprocess::Whiten;
      net._param.outputWhiteningBias = jObj.at("whitening bias");
    }
  }
}


void omnilearn::NetworkIO::loadCoefs(Network& net, json const& jObj)
{
  net._layers.resize(jObj.size());
  for(size_t i = 0; i < net._layers.size(); i++)
  {
    net._layers[i] = jObj.at(i);
  }
}


void omnilearn::NetworkIO::saveTest(Network const& net, json& jObj) const
{
  Matrix testRes(net.process(net._testRawInputs));
  for(size_t i = 0; i < net._outputLabels.size(); i++)
  {
    jObj[i]["label"] = net._outputLabels[i];
    jObj[i]["expected"] = net._testRawOutputs.col(i);
    jObj[i]["predicted"] = testRes.col(i);
  }
}