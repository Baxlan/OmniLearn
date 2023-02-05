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
    throw Exception("The specified separator have not been found in line 1 of " + path.string());

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



  // get the amount of pending lines
  size_t pending = 0;
  for(; true; pending++)
  {
    if(content[content.size()-1 - pending] != "")
      break;
  }



  size_t place = content[0].find(std::string(1,separator)+separator);
  if(place == std::string::npos)
    throw Exception("The label line doesn't have the input/output separator (which should be a double separator)");

  size_t countBefore = std::count(content[0].begin(), content[0].begin()+place, separator);
  size_t countAfter = std::count(content[0].begin()+place, content[0].end(), separator);
  size_t elements = static_cast<size_t>(std::count(content[0].begin(), content[0].end(), separator));
  std::exception_ptr ep = nullptr;
  tasks.resize(content.size()-1-pending);

  // test if all lines have the same amount of separators and if the double ocurrence appears at the same place
  for(size_t i = 1; i < content.size()-pending; i++)
  {
    tasks[i-1] = t.enqueue([i, &content, separator, path, &ep, elements, countBefore, countAfter]()->void
    {
      try
      {
        if(static_cast<size_t>(std::count(content[i].begin(), content[i].end(), separator)) != elements)
          throw Exception("Line " + std::to_string(i+1) + " of file " + path.string() + " has not the same amount of elements as the label line");

        size_t place2 = content[i].find(std::string(1,separator)+separator);
        if(static_cast<size_t>(std::count(content[i].begin(), content[i].begin()+place2, separator)) != countBefore ||
           static_cast<size_t>(std::count(content[i].begin()+place2, content[i].end(), separator)) != countAfter)
          throw Exception("At line " + std::to_string(i+1) + " in " + path.string() + ", the input/output separator is not at the same place as the label line's one");
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



  data.inputs  = Matrix(content.size()-2-pending, data.inputLabels.size());
  data.outputs = Matrix(content.size()-2-pending, data.outputLabels.size());
  tasks.resize(content.size()-2-pending);

  for(i = 0; i < content.size()-2-pending; i++)
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



  // test if all outputs have the same info
  val = data.outputInfos[0];
  for(i = 1; i < data.outputInfos.size(); i++)
  {
    if(val != data.outputInfos[i])
      throw Exception("Either the info line is missing, or dummy and continuous outputs are mixed");
  }



  // test if convolutional inputs are adjacents
  for(i = 1; i < data.inputInfos.size(); i++)
  {
    if(data.inputInfos[i].find("conv") == std::string::npos)
    {
      if(data.inputInfos[i-1].find("conv") != std::string::npos)
        throw Exception("Either all convolutional inputs are not adjacents, or they are not the last inputs");
    }
  }

  return data;
}