// fileString.hh

#ifndef OMNILEARN_FILESTRING_HH_
#define OMNILEARN_FILESTRING_HH_

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>



namespace omnilearn
{



std::string strip(std::string str, char c);
std::vector<std::string> split(std::string str, char c);
std::string removeRepetition(std::string const& str, char c);
std::vector<std::string> readLines(std::string path);
std::vector<std::string> readCleanLines(std::string path);
void writeLines(std::vector<std::string> const& text, std::ostream& stream);


} // namespace omnilearn




#endif // OMNILEARN_FILESTRING_HH_