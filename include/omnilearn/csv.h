// csv.h

#ifndef OMNILEARN_CSV_H_
#define OMNILEARN_CSV_H_

#include <fstream>
#include <filesystem>

#include "Matrix.hh"

namespace fs = std::filesystem;



namespace omnilearn
{



struct Data
{
  Data():
  inputs(),
  outputs(),
  inputLabels(0),
  outputLabels(0),
  inputInfos(0),
  outputInfos(0)
  {}

  Matrix inputs;
  Matrix outputs;
  std::vector<std::string> inputLabels;
  std::vector<std::string> outputLabels;
  std::vector<std::string> inputInfos;
  std::vector<std::string> outputInfos;
};


Data loadData(fs::path const& path, char separator, size_t threads = 1);


} // namespace omnilearn

#endif // OMNILEARN_CSV_H_