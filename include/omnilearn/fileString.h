// fileString.h

#ifndef OMNILEARN_FILESTRING_H_
#define OMNILEARN_FILESTRING_H_

#include <fstream>
#include <string>
#include <vector>
#include <filesystem>



namespace fs = std::filesystem;


namespace omnilearn
{



std::string strip(std::string str, char c);
std::vector<std::string> split(std::string str, char c);
std::string removeDoubles(std::string const& str, char c);
std::string removeOccurences(std::string str, char c);
std::vector<std::string> readLines(fs::path path);
void writeLines(std::vector<std::string> const& text, std::ostream& stream);


} // namespace omnilearn




#endif // OMNILEARN_FILESTRING_H_