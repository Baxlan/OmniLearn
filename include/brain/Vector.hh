#ifndef BRAIN_VECTOR_HH_
#define BRAIN_VECTOR_HH_

#include <vector>

namespace brain
{



class Vector
{
public:
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

  Vector& operator=(std::vector<double> const& vec);
  operator std::vector<double>() const {return _vec;};
  double& at(size_t index);
  double at(size_t index) const;
  double& operator[](size_t index);
  double operator[](size_t index) const;
  auto begin();
  auto end();
  size_t size() const;
  void reserve(size_t size);
  void push_back(double val);
  void pop_back();

  double sum() const;
  double quadraticSum(size_t order = 2) const;
  double absoluteSum() const;

  std::pair<double, double> mean();
  double dot(Vector const& vec) const;
  double distance(Vector const& vec, size_t order = 2) const;
  double distanceInf(Vector const& vec) const;
  double norm(size_t order = 2) const;
  double normInf() const;
  Vector cross(Vector const& vec) const; //vectorial product, dim 3 or 7

private:
  std::vector<double> _vec;
};



Vector& Vector::operator=(std::vector<double> const& vec)
{
  _vec = vec;
  return *this;
}


double& Vector::at(size_t index)
{
  return _vec.at(index);
}


double Vector::at(size_t index) const
{
  return _vec.at(index);
}


double& Vector::operator[](size_t index)
{
  return _vec[index];
}


double Vector::operator[](size_t index) const
{
  return _vec[index];
}


auto Vector::begin()
{
  return _vec.begin();
}


auto Vector::end()
{
  return _vec.end();
}


size_t Vector::size() const
{
  return _vec.size();
}


void Vector::reserve(size_t size)
{
  _vec.reserve(size);
}


void Vector::push_back(double val)
{
  _vec.push_back(val);
}


void Vector::pop_back()
{
  _vec.pop_back();
}


double Vector::sum() const
{
  double result = 0;
  for(double a : _vec)
    result += a;
  return result;
}


double Vector::quadraticSum(size_t order) const
{
  double result = 0;
  for(double a : _vec)
    result += std::pow(a, order);
  return result;
}


double Vector::absoluteSum() const
{
  double result = 0;
  for(double a : _vec)
    result += std::abs(a);
  return result;
}


std::pair<double, double> Vector::mean()
{
  double mean = 0;
  double dev = 0;
  for(double val : _vec)
    mean += val;
  mean /= size();
  for(double val : _vec)
    dev += std::pow(mean - val, 2);
  dev /= (size() - 1);
  dev = std::sqrt(dev);
  return {mean, dev};
}


double Vector::dot(Vector const& vec) const
{
  if(size() != vec.size())
    throw Exception("In a dot product, both vectors must have the same number of element.");
  double result = 0;
  for(size_t i = 0; i < size(); i++)
    result += ((*this)[i] * vec[i]);
  return result;
}


double Vector::distance(Vector const& vec, size_t order) const
{
  if(size() != vec.size())
    throw Exception("In a vector distance calculation, both vectors must have the same number of element.");
  double result = 0;
  for(size_t i = 0; i < size(); i++)
    result += std::pow(std::abs((*this)[i] - vec[i]), order);
  return std::pow(result, 1/order);
}


double Vector::distanceInf(Vector const& vec) const
{
  if(size() != vec.size())
    throw Exception("In a vector distance calculation, both vectors must have the same number of element.");
  double result = 0;
  for(size_t i = 0; i < size(); i++)
  {
    double temp = std::abs((*this)[i] - vec[i]);
    if(temp > result)
      result = temp;
  }
  return result;
}


double Vector::norm(size_t order) const
{
  return distance(Vector(size(), 0), order);
}


double Vector::normInf() const
{
  return distanceInf(Vector(size(), 0));
}


Vector Vector::cross(Vector const& vec) const //vectorial product, dim 3 or 7
{
  if(size() != vec.size())
    throw Exception("In a cross product, both vectors must have the same number of element.");
  if(size() != 3 && size() != 7)
    throw Exception("Cross product only exists in 3D and 7D space.");
  Vector res(3);
  if(size() == 3)
  {

  }
  else if(size() == 7)
  {

  }
  return res;
}



} // namespace brain

#endif // BRAIN_VECTOR_HH_