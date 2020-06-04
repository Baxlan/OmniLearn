// csv.h

#ifndef OMNILEARN_CSV_H_
#define OMNILEARN_CSV_H_

#include <fstream>

#include "Matrix.hh"




namespace omnilearn
{



struct Data
{
  Data():
  inputs(),
  outputs(),
  inputLabels(0),
  outputLabels(0)
  {}

  Matrix inputs;
  Matrix outputs;
  std::vector<std::string> inputLabels;
  std::vector<std::string> outputLabels;
};



Data loadData(std::string const& path, char separator, size_t threads = 1);



} // namespace omnilearn

#endif // OMNILEARN_CSV_H_