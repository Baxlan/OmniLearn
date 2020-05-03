// NetworkIO.hh

#ifndef OMNILEARN_NETWORKIO_HH_
#define OMNILEARN_NETWORKIO_HH_

#include <exception>
#include <fstream>
#include "json.hh"

using json = nlohmann::json;

namespace omnilearn
{



class Network;



class NetworkIO
{
public:
  NetworkIO(std::string const& path);
  void list(std::string const& line); // write in real time what the network does
  void saveNet(Network const& net);   // write the state of the network when it learned
  void saveTest(Network const& net);  // write expected and predicted test data
  void load(Network& net) const;

private:
  void saveParameters(Network const& net, json& jObj);
  void saveInputPreprocess(Network const& net, json& jObj);
  void saveOutputPreprocess(Network const& net, json& jObj);
  void saveCoefs(Network const& net, json& jObj);
  void loadParameters(Network& net, json const& jObj) const;
  void loadInputPreprocess(Network& net, json const& jObj) const;
  void loadOutputPreprocess(Network& net, json const& jObj) const;
  void loadCoefs(Network& net, json const& jObj) const;

private:
  std::string _path;
  std::fstream _listing;
  std::fstream _save;
  std::fstream _test;
};



} // namespace omnilearn



#endif // OMNILEARN_NETWORKIO_HH_