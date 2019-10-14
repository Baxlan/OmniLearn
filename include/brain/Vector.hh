#ifndef BRAIN_VECTOR_HH_
#define BRAIN_VECTOR_HH_

#include <vector>
#include <type_traits>
#include <initializer_list>
#include <cmath>

#include "Exception.hh"

namespace brain
{

class Matrix;

//=============================================================================
//=============================================================================
//=============================================================================
//=== IS_ITERABLE =============================================================
//=============================================================================
//=============================================================================
//=============================================================================



// we consider T is iterable if it has a end() method returning an iterator
template <typename T>
struct is_iterable
{
  template<typename U, typename U::const_iterator (U::*)() const> struct SFINAE {};
  template<typename U> static char Test(SFINAE<U, &U::end>*);
  template<typename U> static int Test(...);
  static const bool value = sizeof(Test<T>(0)) == sizeof(char);
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== VECTOR IMPLEMENTATION ===================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Vector
{
public:
  typedef std::vector<double>::iterator iterator;
  typedef std::vector<double>::const_iterator const_iterator;

  //"default" constructor
  Vector(size_t size = 0, double val = 0):
  _vec(size, val)
  {
  }

  //copy contructor
  Vector(Vector const& vec):
  _vec(vec._vec)
  {
  }

  //constructor taking any STL container
  template <typename T, typename = typename std::enable_if<is_iterable<T>::value>::type, typename = typename std::enable_if<std::is_arithmetic<typename T::value_type>::value>::type>
  Vector(T const& container):
  _vec(container.begin(), container.end())
  {
  }


  //constructor taking an initializer_list to allow brace enclosed initializer list
  Vector(std::initializer_list<double> const& list):
  _vec(list.begin(), list.end())
  {
  }


  //constructor taking range
  template <typename T, typename = typename std::enable_if<std::is_arithmetic<typename std::iterator_traits<T>::value_type>::value>::type>
  Vector(T const& beg, T const& end):
  _vec(beg, end)
  {
  }

  //constructor taking a raw array
  template <typename T, size_t N, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  Vector(T (&table)[N]):
  _vec(std::begin(table), std::end(table))
  {
  }

  template <typename T, typename = typename std::enable_if<is_iterable<T>::value, void>::type, typename = typename std::enable_if<std::is_arithmetic<typename T::value_type>::value>::type>
  operator T() const {return T(_vec.begin(), _vec.end());}

  //assignment operator taking any type
  template <typename T, typename = typename std::enable_if<is_iterable<T>::value, void>::type>
  Vector& operator=(T const& container)
  {
    *this = Vector(container);
    return *this;
  }

  double& at(size_t index);
  double const& at(size_t index) const;
  double& operator[](size_t index);
  double const& operator[](size_t index) const;
  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
  size_t size() const;
  void reserve(size_t size);
  void push_back(double val);
  void pop_back();
  double const* data() const;

  double sum() const;
  double quadraticSum(size_t order = 2) const;
  double absoluteSum() const;
  std::pair<double, double> mean() const; // return mean and deviation
  double norm(size_t order = 2) const;
  double normInf() const;
  std::pair<double, double> minMax() const;

  static double dot(Vector const& a, Vector const& b);
  static double distance(Vector const& a, Vector const& b, size_t order = 2);
  static double distanceInf(Vector const& a, Vector const& b);
  static Vector cross(Vector const& a, Vector const& b);
  static Vector hadamard(Vector a, Vector const& b);
  static Matrix matrix(Vector const& a, Vector const& b); // implemented in Matrix.hh because Matrix implementation is needed

  Vector& operator+=(Vector const& vec);
  Vector& operator-=(Vector const& vec);

  template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  Vector& operator*=(T val)
  {
    for(double& a : _vec)
      a *= val;
    return *this;
  }

  template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  Vector& operator/=(T val)
  {
    for(double& a : _vec)
      a /= val;
    return *this;
  }

private:
  std::vector<double> _vec;
};


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


Vector::iterator Vector::begin()
{
  return _vec.begin();
}


Vector::const_iterator Vector::begin() const
{
  return _vec.begin();
}


Vector::iterator Vector::end()
{
  return _vec.end();
}


Vector::const_iterator Vector::end() const
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


double const* Vector::data() const
{
  return _vec.data();
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


std::pair<double, double> Vector::mean() const
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


std::pair<double, double> Vector::minMax() const
{
  return {*std::min_element(begin(), end()), *std::max_element(begin(), end())};
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
    res[0] = (a[1] * b[2]) - (a[2] * b[1]);
    res[1] = (a[2] * b[0]) - (a[0] * b[2]);
    res[2] = (a[0] * b[1]) - (a[1] * b[0]);
  }
  else //size = 7
  {
    res[0] = (a[1] * b[3]) - (a[3] * b[1]) + (a[2] * b[6]) - (a[6] * b[2]) + (a[4] * b[5]) - (a[5] * b[4]);
    res[1] = (a[2] * b[4]) - (a[4] * b[2]) + (a[3] * b[0]) - (a[0] * b[3]) + (a[5] * b[6]) - (a[6] * b[5]);
    res[2] = (a[3] * b[5]) - (a[5] * b[3]) + (a[4] * b[1]) - (a[1] * b[4]) + (a[6] * b[0]) - (a[0] * b[6]);
    res[3] = (a[4] * b[6]) - (a[6] * b[4]) + (a[5] * b[2]) - (a[2] * b[5]) + (a[0] * b[1]) - (a[1] * b[0]);
    res[4] = (a[5] * b[0]) - (a[0] * b[5]) + (a[6] * b[3]) - (a[3] * b[6]) + (a[1] * b[2]) - (a[2] * b[1]);
    res[5] = (a[6] * b[1]) - (a[1] * b[6]) + (a[0] * b[4]) - (a[4] * b[0]) + (a[2] * b[3]) - (a[3] * b[2]);
    res[6] = (a[0] * b[2]) - (a[2] * b[0]) + (a[1] * b[5]) - (a[5] * b[1]) + (a[3] * b[4]) - (a[4] * b[3]);
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


Vector& Vector::operator+=(Vector const& vec)
{
  if(size() != vec.size())
    throw Exception("To sum two vectors, they must have the same length.");
  for(size_t i = 0; i < size(); i++)
    _vec[i] += vec[i];
  return *this;
}


Vector& Vector::operator-=(Vector const& vec)
{
  if(size() != vec.size())
    throw Exception("To subtract two vectors, they must have the same length.");
  for(size_t i = 0; i < size(); i++)
    _vec[i] -= vec[i];
  return *this;
}


Vector operator+(Vector a, Vector const& b)
{
  return a += b;
}


Vector operator-(Vector a, Vector const& b)
{
  return a -= b;
}


template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
operator*(T val, Vector a)
{
  return a *= val;
}


template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
operator*(Vector a, T val)
{
  return a *= val;
}


template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
operator/(T val, Vector a)
{
  return a /= val;
}


template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
operator/(Vector a, T val)
{
  return a /= val;
}



} // namespace brain

#endif // BRAIN_VECTOR_HH_