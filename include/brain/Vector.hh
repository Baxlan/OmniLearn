#ifndef BRAIN_VECTOR_HH_
#define BRAIN_VECTOR_HH_

#include <vector>

namespace brain
{



class Vector
{
  Vector():
  _vec()
  {
  }

  Vector(size_t size, double val = 0):
  _vec(std::vector<double>(size, val))
  {
  }

  Vector(std::vector<double> const& vec):
  _vec(vec)
  {
  }

  Vector(Vector const& vec):
  _vec(vec._vec)
  {
  }

  operator=(std::vector<double> const& vec);
  auto begin();
  auto end();
  void reserve(size_t size);
  operator std::vector<double>();
  double& at();
  double& operator[]();
private:
  std::vector<double> _vec;
};



} // namespace brain

#endif // BRAIN_VECTOR_HH_