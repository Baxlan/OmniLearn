// NetworkIO.hh

#ifndef OMNILEARN_NETWORKIO_HH_
#define OMNILEARN_NETWORKIO_HH_

#include "json.hh"

#include <iostream>
#include <filesystem>
#include <fstream>



using json = nlohmann::json;
namespace fs = std::filesystem;



namespace omnilearn
{



// first declaration needed because the friend is considered as declaration too,
// but friend declaraton can't take a default template parameter,
// and a declaration AFTER the class is considered as redeclaration of the friend, and there is a mismatch
// because the "friend" one have not the default template parameter.
class NetworkIO;

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
NetworkIO& operator<<(NetworkIO& io, T const& text);



class Network;

class NetworkIO
{
  template<typename T, typename> friend NetworkIO& operator<<(NetworkIO& io, T const& text);
  friend NetworkIO& operator<<(NetworkIO& io, std::string const& text);

public:
  NetworkIO(fs::path const& path, bool verbose);
  void save(Network const& net) const;   // write the state of the network when it learned
  static void load(Network& net, fs::path const& path);

private:
  void saveParameters(Network const& net, json& jObj) const;
  void saveInputPreprocess(Network const& net, json& jObj) const;
  void saveOutputPreprocess(Network const& net, json& jObj) const;
  void saveCoefs(Network const& net, json& jObj) const;
  void saveTest(Network const& net, json& jObj) const;
  static void loadParameters(Network& net, json const& jObj);
  static void loadInputPreprocess(Network& net, json const& jObj);
  static void loadOutputPreprocess(Network& net, json const& jObj);
  static void loadCoefs(Network& net, json const& jObj);

private:
  fs::path _path;
  std::ofstream _listing;
  bool _verbose;
};



template<typename T, typename>
NetworkIO& operator<<(NetworkIO& io, T const& text)
{
  io._listing << text;
  if(io._verbose)
    std::cout << text;
  return io;
}


// not a specialization, but an overload
template<typename T, size_t N>
NetworkIO& operator<<(NetworkIO& io, T text[N])
{
  return (io << std::string(text));
}


// not template, implemented in .cpp
NetworkIO& operator<<(NetworkIO& io, std::string const& text);



} // namespace omnilearn



#endif // OMNILEARN_NETWORKIO_HH_