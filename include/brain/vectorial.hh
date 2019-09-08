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
        throw Exception("In a dot product, both vectors must have the same number of element.");

    double result = 0;
    for(unsigned i = 0; i < a.size(); i++)
    {
        result += (a[i] * b[i]);
    }
    return result;
}


double distance(std::vector<double> const& a, std::vector<double> const& b, double order = 2)
{
    if(a.size() != b.size())
        throw Exception("To calculate the dispance between two vectors, they must have the same number of element.");

    double result = 0;
    for(unsigned i=0; i<a.size(); i++)
    {
        result += std::pow((a[i] - b[i]), order);
    }
    return std::pow(result, 1/order);
}


//first is average, second is deviation
std::pair<double, double> average(std::vector<double> const& vec)
{
    double mean = 0;
    double dev = 0;
    for(double const& val : vec)
    {
        mean += val;
    }
    mean /= vec.size();

    for(double const& val : vec)
    {
        dev += std::pow(mean - val, 2);
    }
    dev /= vec.size();
    dev = std::sqrt(dev);

    return {mean, dev};
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


std::pair<double, double> minMax(std::vector<double> const& vec)
{
    return {*std::min_element(vec.begin(), vec.end()), *std::max_element(vec.begin(), vec.end())};
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


std::vector<double> column(Matrix const& a, unsigned index)
{
    std::vector<double> b(a[0].size(), 0);
    for(unsigned i = 0; i < b.size(); i++)
    {
        b[i] = a[i][index];
    }
    return b;
}

} // namespace brain



#endif // BRAIN_VECTORIAL_HH_