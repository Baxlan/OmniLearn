// csv.cpp

#include "omnilearn/csv.hh"
#include "omnilearn/Exception.hh"
#include "omnilearn/ThreadPool.hh"



omnilearn::Data omnilearn::loadData(std::string const& path, char separator, size_t threads)
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
  if (content[0].find(separator) == std::string::npos)
    throw Exception("Wrong separator used to read csv.");

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