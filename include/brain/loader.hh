#ifndef BRAIN_LOADER_HH_
#define BRAIN_LOADER_HH_

#include "Network.hh"

namespace brain
{



Network load(std::string const& name = "brain_network")
{
  std::ifstream save(name + ".save");
  std::ifstream out(name + ".out");

  if(!save)
    throw Exception("Cannot open " + name + ".save");
  if(!out)
    throw Exception("Cannot open " + name + ".out");
}



} // namespace brain

#endif // BRAIN_LOADER_HH_