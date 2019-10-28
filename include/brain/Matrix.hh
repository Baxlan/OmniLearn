#ifndef BRAIN_MATRIX_HH_
#define BRAIN_MATRIX_HH_

#include "Vector.hh"

namespace brain
{



class Matrix
{
public:
  typedef std::vector<Vector>::iterator iterator;
  typedef std::vector<Vector>::const_iterator const_iterator;

  //"default" constructor
  Matrix(size_t lines = 0, Vector col = {0, 0}):
  _matrix(lines, col)
  {
  }

  //copy constructor
  Matrix(Matrix const& matrix):
  _matrix(matrix._matrix)
  {
  }

  //constructor taking any STL container of container
  template <typename T, typename = typename std::enable_if<is_iterable<T>::value, void>::type, typename = typename std::enable_if<is_iterable<typename T::value_type>::value, void>::type, typename = typename std::enable_if<std::is_arithmetic<typename T::value_type::value_type>::value>::type>
  Matrix(T const& container):
  _matrix(container.begin(), container.end())
  {
  }

  //constructor taking an initializer_list to allow brace enclosed initializer list
  Matrix(std::initializer_list<Vector> const& list):
  _matrix(list.begin(), list.end())
  {
  }

  //constructor taking range
  template <typename T, typename = typename std::enable_if<is_iterable<typename std::iterator_traits<T>::value_type>::value>::type, typename = typename std::enable_if<std::is_arithmetic<typename std::iterator_traits<T>::value_type::value_type>::value>::type>
  Matrix(T const& beg, T const& end):
  _matrix(beg, end)
  {
  }

  //constructor taking a raw 2D array of double
  template <typename T, size_t N, size_t M, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  Matrix(T (&table)[N][M]):
  _matrix(std::begin(table), std::end(table))
  {
  }

  //assignment operator taking any type
  template <typename T, typename = typename std::enable_if<is_iterable<T>::value, void>::type>
  Matrix& operator=(T const& container)
  {
    *this = Matrix(container);
    return *this;
  }

  Vector& at(size_t line);
  Vector const& at(size_t line) const;
  Vector& operator[](size_t line);
  Vector const& operator[](size_t line) const;
  Vector const column(size_t col) const;
  void column(size_t col, Vector const& vec);
  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
  size_t lines() const;
  size_t columns() const;
  void resizeLines(size_t lines);
  void resizeColumns(size_t col);
  void addLine(Vector const& line);
  void addColumn(Vector const& col);
  void popLine();
  void popColumn();
  std::vector<Vector> data() const;

  void transpose();
  static Matrix transpose(Matrix const& matrix);
  static Matrix hadamard(Matrix a, Matrix const& b);

  Matrix& operator+=(Matrix const& a);
  Matrix& operator-=(Matrix const& a);
  Matrix& operator*=(Matrix const& a);
  template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  Matrix& operator*=(T const& val)
  {

  }

private:
  void check() const; //throw if all lines have not the same length

private:
  std::vector<Vector> _matrix;
};


Vector& Matrix::at(size_t line)
{
  return _matrix.at(line);
}


Vector const& Matrix::at(size_t line) const
{
  return _matrix.at(line);
}


Vector& Matrix::operator[](size_t line)
{
  return _matrix[line];
}


Vector const&  Matrix::operator[](size_t line) const
{
  return _matrix[line];
}


Vector const Matrix::column(size_t col) const
{
  Vector vec(lines(), 0);
  for(size_t i = 0; i < lines(); i++)
  {
    vec[i] = _matrix[i][col];
  }
  return vec;
}


void Matrix::column(size_t col, Vector const& vec)
{
  if(vec.size() != lines())
    throw Exception("The length of the column must be the same as the number of lines of the matrix.");
  for(size_t i = 0; i < lines(); i++)
  {
    _matrix[i][col] = vec[i];
  }
}


Matrix::iterator Matrix::begin()
{
  return _matrix.begin();
}


Matrix::const_iterator Matrix::begin() const
{
  return _matrix.begin();
}


Matrix::iterator Matrix::end()
{
  return _matrix.end();
}


Matrix::const_iterator Matrix::end() const
{
  return _matrix.end();
}


size_t Matrix::lines() const
{
  return _matrix.size();
}


size_t Matrix::columns() const
{
  if(lines() == 0)
    return 0;
  else
    return _matrix[0].size();
}


void Matrix::resizeLines(size_t lines)
{
  _matrix.resize(lines);
}


void Matrix::resizeColumns(size_t col)
{
  for(Vector& vec : _matrix)
    vec.resize(col);
}


void Matrix::addLine(Vector const& line)
{
  _matrix.push_back(line);
}


void Matrix::addColumn(Vector const& col)
{
  if(col.size() != lines())
    throw Exception("The length of the column must be the same as the number of lines of the matrix.");
  for(size_t i = 0; i < lines(); i++)
    _matrix[i].push_back(col[i]);
}


void Matrix::popLine()
{
  _matrix.pop_back();
}


void Matrix::popColumn()
{
  for(Vector& a: _matrix)
    a.pop_back();
}


std::vector<Vector> Matrix::data() const
{
  return _matrix;
}


void Matrix::transpose()
{
  *this = transpose(*this);
}


Matrix Matrix::transpose(Matrix const& matrix)
{
  Matrix returnMatrix(matrix.columns(), Vector(matrix.lines(), 0));
  for(size_t i = 0; i < matrix.lines(); i++)
  {
    for(size_t j = 0; j < matrix[0].size(); j++)
    {
      returnMatrix[j][i] = matrix[i][j];
    }
  }
  return returnMatrix;
}


Matrix Matrix::hadamard(Matrix a, Matrix const& b)
{
  if(a.lines() != b.lines())
    throw Exception("To perform hadamard product on two matrices, they must have the same dimensions.");
  else if(a.columns() != b.columns())
    throw Exception("To perform hadamard product on two matrices, they must have the same dimensions.");
  for(size_t i = 0; i < a.lines(); i++)
  {
    for(size_t j = 0; j < a.columns(); j++)
    {
      a[i][j] *= b[i][j];
    }
  }
  return a;
}


Matrix& Matrix::operator+=(Matrix const& a)
{
  if(lines() != a.lines())
    throw Exception("To sum two matrices, they must have the same dimensions.");
  else if(columns() != a.columns())
    throw Exception("To sum two matrices, they must have the same dimensions.");
  for(size_t i = 0; i < lines(); i++)
  {
    for(size_t j = 0; j < columns(); j++)
    {
      _matrix[i][j] += a[i][j];
    }
  }
  return *this;
}


Matrix& Matrix::operator-=(Matrix const& a)
{
  if(lines() != a.lines())
    throw Exception("To subtract two matrices, they must have the same dimensions.");
  else if(columns() != a.columns())
    throw Exception("To subtract two matrices, they must have the same dimensions.");
  for(size_t i = 0; i < lines(); i++)
  {
    for(size_t j = 0; j < columns(); j++)
    {
      _matrix[i][j] -= a[i][j];
    }
  }
  return *this;
}


Matrix& Matrix::operator*=(Matrix const& a)
{
  if(columns() != a.lines())
    throw Exception("To multiply two matrices, the number of columns of the first matrix must be the number of lines of the second matrix.");
  Matrix product(lines(), Vector(a.columns(), 0));
  for(size_t i = 0; i < lines(); i++)
  {
    for(size_t j = 0; j < a.columns(); j++)
    {
      product[i][j] = Vector::dot(_matrix[i], a.column(j));
    }
  }
  *this = product;
  return *this;
}


Matrix operator+(Matrix a, Matrix const& b)
{
  return a += b;
}


Matrix operator-(Matrix a, Matrix const& b)
{
  return a -= b;
}


Matrix operator*(Matrix a, Matrix const& b)
{
  return a *= b;
}


template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
Matrix operator*(Matrix a, T val)
{
  return a *= val;
}


template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
Matrix operator*(T val, Matrix a)
{
  return a *= val;
}


void Matrix::check() const
{
  if(lines() != 0)
  {
    size_t size = _matrix[0].size();
    for(size_t i = 0; i < lines(); i++)
      if(_matrix[i].size() != size)
        throw Exception("All lines of a matrix must have the same length.");
  }
}


//return a Matrix
Matrix Vector::matrix(Vector const& a, Vector const& b)
{
  Matrix matrix(a.size(), Vector(b.size(), 0));
  for(size_t i = 0; i < a.size(); i++)
  {
    for(size_t j = 0; j < b.size(); j++)
    {
      matrix[i][j] = a[i]*b[j];
    }
  }
  return matrix;
}



} // namespace brain

#endif // BRAIN_MATRIX_HH_