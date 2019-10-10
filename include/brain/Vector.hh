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

  //constructor taking any STL container
  template <typename container_t>
  Vector(container_t const& container):
  _vec(container.begin(), container.end())
  {
  }

  //copy contructor
  Vector(Vector const& vec):
  _vec(vec._vec)
  {
  }

  //constructor taking a raw array of double
  template <size_t N>
  Vector(double (&table)[N]):
  _vec(std::begin(table), std::end(table))
  {
  }

  Vector& operator=(std::vector<double> const& vec);
  template <typename T> operator T() const {return T(_vec.begin(), _vec.end());};
  double& at(size_t index);
  double const& at(size_t index) const;
  double& operator[](size_t index);
  double const& operator[](size_t index) const;
  std::vector<double>::iterator begin();
  std::vector<double>::const_iterator begin() const;
  std::vector<double>::iterator end();
  std::vector<double>::const_iterator end() const;
  size_t size() const;
  void reserve(size_t size);
  void push_back(double val);
  void pop_back();

  double sum() const;
  double quadraticSum(size_t order = 2) const;
  double absoluteSum() const;
  std::pair<double, double> mean(); // return mean and deviation
  double norm(size_t order = 2) const;
  double normInf() const;

  static double dot(Vector const& a, Vector const& b);
  static double distance(Vector const& a, Vector const& b, size_t order = 2);
  static double distanceInf(Vector const& a, Vector const& b);
  static Vector cross(Vector const& a, Vector const& b);
  static Vector hadamard(Vector a, Vector const& b);
  static Vector matrix(Vector const& a, Vector const& b);

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


double const& Vector::at(size_t index) const
{
  return _vec.at(index);
}


double& Vector::operator[](size_t index)
{
  return _vec[index];
}


double const& Vector::operator[](size_t index) const
{
  return _vec[index];
}


std::vector<double>::iterator Vector::begin()
{
  return _vec.begin();
}


std::vector<double>::const_iterator Vector::begin() const
{
  return _vec.begin();
}


std::vector<double>::iterator Vector::end()
{
  return _vec.end();
}


std::vector<double>::const_iterator Vector::end() const
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
  double mean = sum() / size();
  double dev = 0;
  for(double val : _vec)
    dev += std::pow(mean - val, 2);
  dev /= (size() - 1);
  dev = std::sqrt(dev);
  return {mean, dev};
}


double Vector::norm(size_t order) const
{
  return distance(*this, Vector(size(), 0), order);
}


double Vector::normInf() const
{
  return distanceInf(*this, Vector(size(), 0));
}


double Vector::dot(Vector const& a, Vector const& b)
{
  if(a.size() != b.size())
    throw Exception("In a dot product, both vectors must have the same number of element.");
  double result = 0;
  for(size_t i = 0; i < a.size(); i++)
    result += (a[i] * b[i]);
  return result;
}


double Vector::distance(Vector const& a, Vector const& b, size_t order)
{
  if(a.size() != b.size())
    throw Exception("In a vector distance calculation, both vectors must have the same number of element.");
  double result = 0;
  for(size_t i = 0; i < a.size(); i++)
    result += std::pow(std::abs(a[i] - b[i]), order);
  return std::pow(result, 1/order);
}


double Vector::distanceInf(Vector const& a, Vector const& b)
{
  if(a.size() != b.size())
    throw Exception("In a vector distance calculation, both vectors must have the same number of element.");
  double result = 0;
  for(size_t i = 0; i < a.size(); i++)
  {
    double temp = std::abs(a[i] - b[i]);
    if(temp > result)
      result = temp;
  }
  return result;
}


Vector Vector::cross(Vector const& a, Vector const& b)
{
  if(a.size() != b.size())
    throw Exception("In a cross product, both vectors must have the same number of element.");
  if(a.size() != 3 && a.size() != 7)
    throw Exception("Cross product only exists in 3D and 7D space.");
  Vector res(a.size());
  if(a.size() == 3)
  {

  }
  else //size = 7
  {

  }
  return res;
}


Vector Vector::hadamard(Vector a, Vector const& b)
{
  if(a.size() != b.size())
    throw Exception("In a hadamard product, both vectors must have the same number of element.");
  for(size_t i = 0; i < a.size(); i++)
    a[i] *= b[i];
  return a;
}


Vector Vector::matrix(Vector const& a, Vector const& b)
{

}



} // namespace brain

#endif // BRAIN_VECTOR_HH_