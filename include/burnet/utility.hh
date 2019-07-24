#ifndef BURNET_UTILITY_HH_
#define BURNET_UTILITY_HH_

#include <algorithm>
#include <chrono>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace burnet
{



typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<std::vector<std::vector<double>>> Tensor;
typedef std::vector<std::pair<std::vector<double>, std::vector<double>>> Dataset;

enum class Distrib {Uniform, Normal};
enum class Loss {L1, L2, Hinge, Hinge2, Entropy};


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


Matrix softmax(Matrix inputs)
{
    for(unsigned i = 0; i < inputs.size(); i++)
    {
        double c = *std::max_element(inputs[i].begin(), inputs[i].end());
        double sum = 0;
        for(unsigned j = 0; j < inputs[0].size(); j++)
        {
            sum += std::exp(inputs[i][j] - c);
        }
        for(unsigned j = 0; j < inputs[0].size(); j++)
        {
            inputs[i][j] = std::exp(inputs[i][j] - c) / sum;
        }
    }
    return inputs;
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
    maxNorm(32767),
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
    learningRate(0.005),
    L1(0),
    L2(0),
    tackOn(0),
    maxEpoch(500),
    epochAfterOptimal(100),
    dropout(0),
    dropconnect(0),
    validationRatio(0.2),
    testRatio(0.2),
    loss(Loss::L2)
    {
    }

    unsigned dataSeed;
    unsigned batchSize;
    double learningRate;
    double L1;
    double L2;
    double tackOn;
    unsigned maxEpoch;
    unsigned epochAfterOptimal;
    double dropout;
    double dropconnect;
    double validationRatio;
    double testRatio;
    Loss loss;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== COST FUNCTIONS ==========================================================
//=============================================================================
//=============================================================================
//=============================================================================


// one line = one feature, one colums = one class
// first are loss, second are gradients
std::pair<Matrix, Matrix> L1Loss(Matrix const& real, Matrix const& predicted)
{
    Matrix loss(real.size(), std::vector<double>(real[0].size(), 0));
    Matrix gradients(real.size(), std::vector<double>(real[0].size(), 0));
    for(unsigned i = 0; i < loss.size(); i++)
    {
        for(unsigned j = 0; j < loss[0].size(); j++)
        {
            loss[i][j] = std::abs(real[i][j] - predicted[i][j]);
            if (real[i][j] < predicted[i][j])
                gradients[i][j] = -1;
            else if (real[i][j] > predicted[i][j])
                gradients[i][j] = 1;
            else
                gradients[i][j] = 0;
        }
    }
    return {loss, gradients};
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
std::pair<Matrix, Matrix> L2Loss(Matrix const& real, Matrix const& predicted)
{
    Matrix loss(real.size(), std::vector<double>(real[0].size(), 0));
    Matrix gradients(real.size(), std::vector<double>(real[0].size(), 0));
    for(unsigned i = 0; i < loss.size(); i++)
    {
        for(unsigned j = 0; j < loss[0].size(); j++)
        {
            loss[i][j] = 0.5 * std::pow(real[i][j] - predicted[i][j], 2);
            gradients[i][j] = -(real[i][j] - predicted[i][j]);
        }
    }
    return  {loss, gradients};
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
std::pair<Matrix, Matrix> entropyLoss(Matrix const& real, Matrix const& predicted)
{
    Matrix softMax = softmax(predicted);
    Matrix loss(real.size(), std::vector<double>(real[0].size(), 0));
    Matrix gradients(real.size(), std::vector<double>(real[0].size(), 0));
    for(unsigned i = 0; i < loss.size(); i++)
    {
        for(unsigned j = 0; j < loss[0].size(); j++)
        {
            loss[i][j] = real[i][j] * -std::log(softMax[i][j]);
            gradients[i][j] = softMax[i][j] - real[i][j];
        }
    }
    return  {loss, gradients};
}

} //namespace burnet

#endif //BURNET_UTILITY_HH_