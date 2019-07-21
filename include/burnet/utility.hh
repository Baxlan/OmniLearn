#ifndef BURNET_UTILITY_HH_
#define BURNET_UTILITY_HH_

#include <chrono>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace burnet
{



typedef std::vector<std::vector<double>> Matrix;
enum class Distrib {Uniform, Normal};
typedef std::vector<std::pair<std::vector<double>, std::vector<double>>> Dataset;


//=============================================================================
//=============================================================================
//=============================================================================
//=== EXCEPTIONS DEFINITION ===================================================
//=============================================================================
//=============================================================================
//=============================================================================



struct Exception : public std::exception
{
    Exception(std::string const& msg):
    _msg("[EasyLearn.exception : " + msg + "]")
    {
    }

    virtual ~Exception()
    {
    }

    virtual const char* what() const noexcept
    {
        return _msg.c_str();
    }

private:
    std::string const _msg;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== UTILITY FUNCTIONS =======================================================
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


double distance(std::vector<double> const& a, std::vector<double> const& b, double order = 2)
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
//=== HYPERPARAMETERS =========================================================
//=============================================================================
//=============================================================================
//=============================================================================



struct LayerParam
{
    LayerParam():
    size(8),
    maxNorm(0),
    distrib(Distrib::Normal),
    distribVal1(distrib == Distrib::Normal ? 0 : 6),
    distribVal2(2),
    k(1)
    {
    }

    unsigned size; //number of neurons
    double maxNorm;
    Distrib distrib;
    double distribVal1; //mean (if uniform), boundary (if uniform)
    double distribVal2; //deviation (if normal) or useless (if uniform)
    unsigned k; //number of weight set for each neuron
};


struct NetworkParam
{
    NetworkParam():
    dataSeed(0),
    batchSize(1),
    learningRate(0.0005),
    maxEpoch(1500),
    epochAfterOptimal(100),
    dropout(0.2),
    dropconnect(0),
    validationRatio(0.2),
    testRatio(0.2)
    {
    }

    unsigned dataSeed;
    unsigned batchSize;
    double learningRate;
    unsigned maxEpoch;
    unsigned epochAfterOptimal;
    double dropout;
    double dropconnect;
    double validationRatio;
    double testRatio;
};



} //namespace burnet

#endif //BURNET_UTILITY_HH_