#ifndef BRAIN_VECTORIAL_HH_
#define BRAIN_VECTORIAL_HH_

#include <vector>
#include "exception.hh"

namespace brain
{



typedef std::vector<std::vector<double>> Matrix;



//=============================================================================
//=============================================================================
//=============================================================================
//=== FUNCTIONS RELATED TO VECTORS ============================================
//=============================================================================
//=============================================================================
//=============================================================================



double dot(std::vector<double> const& a, std::vector<double> const& b)
{
    if(a.size() != b.size())
    {
        throw Exception("In a dot product, both vectors must have the same number of element.");
    }

    double result = 0;
    for(unsigned i = 0; i < a.size(); i++)
    {
        result += (a[i] * b[i]);
    }
    return result;
}


double distance(std::vector<double> const& a, std::vector<double> const& b, double order)
{
    if(a.size() != b.size())
    {
        throw Exception("To calculate the dispance between two vectors, they must have the same number of element.");
    }

    double result = 0;
    for(unsigned i=0; i<a.size(); i++)
    {
        result += std::pow((a[i] - b[i]), order);
    }
    return std::pow(result, 1/order);
}


double average(std::vector<double> const& a)
{
    double sum = 0;
    for(double const& val : a)
    {
        sum += val;
    }
    return sum / a.size();
}


double sum(std::vector<double> const& vec)
{
    double result = 0;
    for(double a : vec)
        result += std::abs(a);
    return result;
}


double absoluteSum(std::vector<double> const& vec)
{
    double result = 0;
    for(double a : vec)
        result += std::abs(a);
    return result;
}


double quadraticSum(std::vector<double> const& vec)
{
    double result = 0;
    for(double a : vec)
        result += std::pow(a, 2);
    return result;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== FUNCTIONS RELATED TO MATRIX =============================================
//=============================================================================
//=============================================================================
//=============================================================================



Matrix transpose(Matrix const& a)
{
    Matrix b(a[0].size(), std::vector<double>(a.size(), 0));
    for(unsigned i = 0; i < a.size(); i++)
    {
        for(unsigned j = 0; j < a[0].size(); j++)
        {
            b[j][i] = a[i][j];
        }
    }
    return b;
}

} // namespace brain



#endif // BRAIN_VECTORIAL_HH_