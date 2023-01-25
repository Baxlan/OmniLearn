// fileString.cpp

#include "omnilearn/fileString.h"
#include "omnilearn/Exception.hh"

#include <algorithm>



std::string omnilearn::strip(std::string str, char c)
{
  while(str[0]==c)
      str.erase(0, 1);
  while(str[str.size()-1]==c)
    str.erase(str.size()-1, 1);
  return str;
}


std::vector<std::string> omnilearn::split(std::string str, char c)
{
  std::vector<std::string> vec(std::count(str.begin(), str.end(), c)+1);
  for(size_t i = 0; i < vec.size(); i++)
  {
    vec[i] = str.substr(0, str.find_first_of(c));
    str.erase(0, str.find_first_of(c)+1);
  }
  vec[vec.size()-1] = str.substr(0);
  return vec;
}


std::string omnilearn::removeDoubles(std::string const& str, char c)
{
  std::string output;
  size_t pos = 0;
  while(true)
  {
    if(pos < str.size()-1 && str[pos] == str[pos+1] && str[pos] == c)
      ;
    else if(pos < str.size())
      output += str[pos];
    else
      break;
    pos++;
  }
  return output;
}


std::string omnilearn::removeOccurences(std::string str, char c)
{
  str.erase(std::remove(str.begin(), str.end(), c), str.end());
  return str;
}


std::vector<std::string> omnilearn::readLines(fs::path path)
{
  std::ifstream file(path);
  if(!file)
    throw Exception("Cannot open " + path.string());
  std::vector<std::string> content;
  while(file.good())
  {
    content.push_back("");
    std::getline(file, content[content.size() - 1]);
  }
  return content;
}


void omnilearn::writeLines(std::vector<std::string> const& text, std::ostream& stream)
{
  for(size_t i = 0; i < text.size(); i++)
    stream << text[i] << "\n";
}