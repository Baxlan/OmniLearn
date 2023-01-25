// csv.cpp

#include "omnilearn/csv.h"
#include "omnilearn/Exception.hh"
#include "omnilearn/ThreadPool.hh"
#include "omnilearn/fileString.h"

namespace fs = std::filesystem;



omnilearn::Data omnilearn::loadData(fs::path const& path, char separator, size_t threads)
{
  std::vector<std::string> content = readLines(path);

  if (content[0].find(separator) == std::string::npos)
    throw Exception("Wrong separator specified to read " + path.string());

  ThreadPool t(threads);
  std::vector<std::future<void>> tasks(content.size()-1);

  // cleaning lines
  for(size_t i = 1; i < content.size(); i++) // start at 1 to avoid modifying labels
  {
    tasks[i-1] = t.enqueue([i, &content, separator]()->void
    {
      content[i] = removeOccurences(content[i], ' ');
      content[i] = removeOccurences(content[i], '\t');
      content[i] = strip(content[i], separator);
    });
  }
  for(size_t  i = 0; i < tasks.size(); i++)
    tasks[i].get();



  content[0] = strip(content[0], separator);
  size_t elements = static_cast<size_t>(std::count(content[0].begin(), content[0].end(), separator));
  std::exception_ptr ep = nullptr;

  // test if all lines have the same amount of separators
  for(size_t i = 1; i < content.size(); i++)
  {
    tasks[i-1] = t.enqueue([i, &content, separator, path, &ep, elements]()->void
    {
      try
      {
        if(static_cast<size_t>(std::count(content[i].begin(), content[i].end(), separator)) != elements)
          throw Exception("Line " + std::to_string(i+1) + " of file " + path.string() + "has not the same amount of elements as the label line");
      }
      catch(...)
      {
          ep = std::current_exception();
      }
    });
  }
  for(size_t  i = 0; i < tasks.size(); i++)
    tasks[i].get();
  if(ep)
        std::rethrow_exception(ep);



  Data data;
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

  //extract inputs infos
  for(i = 0; i < elements; i++)
  {
    val = content[1].substr(0, content[1].find(separator));
    if(val == "")
      break;
    data.inputInfos.push_back(val);
    content[1].erase(0, content[1].find(separator) + 1);
  }
  content[1].erase(0, content[1].find(separator) + 1);

  //extract output infos
  for(; i < elements; i++)
  {
    val = content[1].substr(0, content[1].find(separator));
    data.outputInfos.push_back(val);
    content[1].erase(0, content[1].find(separator) + 1);
  }



  data.inputs  = Matrix(content.size()-2, data.inputLabels.size());
  data.outputs = Matrix(content.size()-2, data.outputLabels.size());
  tasks.resize(content.size()-2);

  for(i = 0; i < content.size()-2; i++)
  {
    tasks[i] = t.enqueue([i, &content, &data, separator]()->void
    {
      std::string line = content[i+2]; // not to read labels nor infos

      // reading inputs
      for(size_t col = 0; col < data.inputLabels.size(); col++)
      {
        data.inputs(i,col) = (std::stod(line.substr(0, line.find(separator))));
        line.erase(0, line.find(separator) + 1);
      }
      line.erase(0, line.find(separator) + 1);

      // reading outputs
      for(size_t col = 0; col < data.outputLabels.size(); col++)
      {
        data.outputs(i,col) = (std::stod(line.substr(0, line.find(separator))));
        line.erase(0, line.find(separator) + 1);
      }
    });
  }
  for(i = 0; i < tasks.size(); i++)
    tasks[i].get();

  return data;
}