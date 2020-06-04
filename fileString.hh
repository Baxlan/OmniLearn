// fileString.h

#ifndef OMNILEARN_FILESTRING_H_
#define OMNILEARN_FILESTRING_H_

#include <fstream>
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




#endif // OMNILEARN_FILESTRING_H_