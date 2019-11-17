#ifndef BRAIN_CSV_HH_
#define BRAIN_CSV_HH_

#include <fstream>

#include "Matrix.hh"

namespace brain
{



//=============================================================================
//=============================================================================
//=============================================================================
//=== DATA STRUCT =============================================================
//=============================================================================
//=============================================================================
//=============================================================================


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



//=============================================================================
//=============================================================================
//=============================================================================
//=== LOAD DATA FUNCTION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================


Data loadData(std::string const& path, char separator, size_t threads = 1)
{
  std::fstream dataFile(path);
  std::vector<std::string> content;
  Data data;

  //load data
  while(dataFile.good())
  {
    content.push_back("");
    std::getline(dataFile, content[content.size()-1]);
  }

  size_t elements = static_cast<size_t>(std::count(content[0].begin(), content[0].end(), separator));
  size_t i = 0;

  //extract inputs labels
  std::string val;
  for(i = 0; i < elements; i++)
  {
    val = content[0].substr(0, content[0].find(separator));
    if(val == "")
      break;
    data.inputLabels.push_back(val);
    content[0].erase(0, content[0].find(separator) + 1);
  }
  content[0].erase(0, content[0].find(separator) + 1);

  //extract output labels
  for(; i < elements; i++)
  {
    val = content[0].substr(0, content[0].find(separator));
    data.outputLabels.push_back(val);
    content[0].erase(0, content[0].find(separator) + 1);
  }

  data.inputs  = Matrix(content.size()-1, data.inputLabels.size());
  data.outputs = Matrix(content.size()-1, data.outputLabels.size());

  ThreadPool t(threads);
  std::vector<std::future<void>> tasks(content.size()-1);

  for(size_t j = 0; j < content.size()-1; j++)
  {
    tasks[j] = t.enqueue([j, &content, &data, separator]()->void
    {
      std::string line = content[j+1]; // do not read the label line
      for(size_t col = 0; col < data.inputLabels.size(); col++)
      {
        data.inputs(j,col) = (std::stod(line.substr(0, line.find(separator))));
        line.erase(0, line.find(separator) + 1);
      }
      line.erase(0, line.find(separator) + 1);
      for(size_t col = 0; col < data.outputLabels.size(); col++)
      {
        data.outputs(j,col) = (std::stod(line.substr(0, line.find(separator))));
        line.erase(0, line.find(separator) + 1);
      }
    });
  }
  for(i = 0; i < tasks.size(); i++)
    tasks[i].get();

  return data;
}



} // namespace brain

#endif // BRAIN_CSV_HH_