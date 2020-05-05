// fileString.cpp

#include "omnilearn/fileString.hh"



std::string omnilearn::strip(std::string str, char c)
{
  while(true)
  {
    if(str[0]==c)
      str.erase(0, 1);
    else
      break;
  }
  while(true)
  {
    if(str[str.size()-1]==c)
      str.erase(str.size()-1, 1);
    else
      break;
  }
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


std::string omnilearn::removeRepetition(std::string const& str, char c)
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


std::vector<std::string> omnilearn::readLines(std::string path)
{
  std::ifstream file(path);
  if(!file)
    throw std::runtime_error("Cannot open " + path);
  std::vector<std::string> content;
  while(file.good())
  {
    content.push_back("");
    std::getline(file, content[content.size() - 1]);
  }
  return content;
}


std::vector<std::string> omnilearn::readCleanLines(std::string path)
{
  std::vector<std::string> content(readLines(path));
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


void omnilearn::writeLines(std::vector<std::string> const& text, std::ostream& stream)
{
  for(size_t i = 0; i < text.size(); i++)
    stream << text[i] << "\n";
}