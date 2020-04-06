// fileString.cpp

#include "omnilearn/fileString.hh"



std::string omnilearn::strip(std::string str, char c)
{
  while(true)
  {
    if(str[0]==c)
      str.erase(0);
    else
      break;
  }
  while(true)
  {
    if(str[str.size()-1]==c)
      str.erase(str.size()-1);
    else
      break;
  }
  return str;
}


std::vector<std::string> omnilearn::split(std::string str, char c)
{
  std::vector<std::string> vec(std::count(str.begin(), str.end(), c));
  for(size_t i = 0; i < str.size()-1; i++)
  {
    vec[i] = str.substr(0, str.find_first_of(c));
    str.erase(0, str.find_first_of(c));
  }
  return vec;
}


std::string omnilearn::removeRepetition(std::string const& str, char c)
{
  std::string output;
  size_t pos = 0;
  while(true)
  {
    if(pos < str.size()-1 && str[pos] == str[pos+1] && str[pos] == c)
      ;
    else
      output += str[pos];
    pos++;
  }
  return output;
}


std::vector<std::string> omnilearn::readLines(std::string path)
{
  std::ifstream file(path);
  if(!file)
    throw std::runtime_error("Cannot open " + path);
  std::istream_iterator<std::string> start(file), end;
  std::vector<std::string> content(start, end);
  return content;
}


std::vector<std::string> omnilearn::readCleanLines(std::string path)
{
  std::ifstream file(path);
  if(!file)
    throw std::runtime_error("Cannot open " + path);
  std::istream_iterator<std::string> start(file), end;
  std::vector<std::string> content(start, end);
  for(size_t i = 0; i < content.size(); i++)
  {
    content[i] = strip(content[i], ' ');
    content[i] = strip(content[i], '\t');
    content[i] = strip(content[i], ' ');
    content[i] = removeRepetition(content[i], ' ');
    content[i] = removeRepetition(content[i], '\t');
  }
  return content;
}