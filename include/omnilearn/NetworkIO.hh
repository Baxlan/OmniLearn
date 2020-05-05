// NetworkIO.hh

#ifndef OMNILEARN_NETWORKIO_HH_
#define OMNILEARN_NETWORKIO_HH_

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
  void saveList() const;
  void saveNet(Network const& net) const;   // write the state of the network when it learned
  void saveTest(Network const& net) const;  // write expected and predicted test data
  void load(Network& net) const;

private:
  void saveParameters(Network const& net, json& jObj) const;
  void saveInputPreprocess(Network const& net, json& jObj) const;
  void saveOutputPreprocess(Network const& net, json& jObj) const;
  void saveCoefs(Network const& net, json& jObj) const;
  void loadParameters(Network& net, json const& jObj) const;
  void loadInputPreprocess(Network& net, json const& jObj) const;
  void loadOutputPreprocess(Network& net, json const& jObj) const;
  void loadCoefs(Network& net, json const& jObj) const;

private:
  std::string _path;
  std::vector<std::string> _listing;
};



} // namespace omnilearn



#endif // OMNILEARN_NETWORKIO_HH_