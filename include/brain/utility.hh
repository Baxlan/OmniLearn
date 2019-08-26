#ifndef BRAIN_UTILITY_HH_
#define BRAIN_UTILITY_HH_

#include <algorithm>
#include <chrono>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace brain
{


typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<std::pair<std::vector<double>, std::vector<double>>> Dataset;

enum class Distrib {Uniform, Normal};
enum class Loss {L1, L2, CrossEntropy};

static inline unsigned nthreads = 1;

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
//=== UTILITY FUNCTIONS ON VECTORS AND MATRIX =================================
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


// the inputs are loss, the output is average loss
double averageLoss(Matrix const& loss)
{
    std::vector<double> feature(loss.size());
    for(unsigned i = 0; i < loss.size(); i++)
    {
        for(unsigned j = 0; j < loss[0].size(); j++)
        {
            feature[i] += loss[i][j];
        }
    }
    double average = 0;
    for(double i : feature)
        average += (i/feature.size());
    return average;
}


double accuracy(Matrix const& real, Matrix const& predicted, double margin)
{
    double unvalidated = 0;
    for(unsigned i = 0; i < real.size(); i++)
    {
        for(unsigned j = 0; j < real[0].size(); j++)
        {
            if(real[i][j] > std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) < predicted[i][j] || predicted[i][j] < (real[i][j] * (1-margin/100)))
                {
                    unvalidated++;
                    break;
                }
            }
            else if(real[i][j] < -std::numeric_limits<double>::epsilon())
            {
                if((real[i][j] * (1+margin/100)) > predicted[i][j] || predicted[i][j] > (real[i][j] * (1-margin/100)))
                {
                    unvalidated++;
                    break;
                }
            }
            else //if real == 0
            {
                if(false)
                {
                    unvalidated++;
                    break;
                }
            }
        }
    }
    return 100* (real.size() - unvalidated)/real.size();
}


//=============================================================================
//=============================================================================
//=============================================================================
//=== LEARNING RATE ANNEALING =================================================
//=============================================================================
//=============================================================================
//=============================================================================

namespace LRDecay
{


    double none(double initialLR, [[maybe_unused]] unsigned epoch, [[maybe_unused]] double decayConstant, [[maybe_unused]] unsigned step)
    {
        return initialLR;
    }


    double inverse(double initialLR, unsigned epoch, double decayConstant, [[maybe_unused]] unsigned step)
    {
        return initialLR / (1 + (decayConstant * (epoch-1)));
    }


    double exp(double initialLR, unsigned epoch, double decayConstant, [[maybe_unused]] unsigned step)
    {
        return initialLR * std::exp(-decayConstant * (epoch-1));
    }


    double step(double initialLR, unsigned epoch, double decayConstant, unsigned step)
    {
        return initialLR * std::pow(decayConstant, std::floor((epoch-1)/step));
    }


} // namespace LRDecay


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
    maxNorm(32),
    distrib(Distrib::Normal),
    mean_boundary(distrib == Distrib::Normal ? 0 : 6),
    deviation(2),
    k(1)
    {
    }

    unsigned size; //number of neurons
    double maxNorm;
    Distrib distrib;
    double mean_boundary; //mean (if uniform), boundary (if uniform)
    double deviation; //deviation (if normal) or useless (if uniform)
    unsigned k; //number of weight set for each neuron
};


struct NetworkParam
{
    NetworkParam():
    seed(0),
    batchSize(1),
    learningRate(0.001),
    L1(0),
    L2(0),
    maxEpoch(500),
    epochAfterOptimal(100),
    dropout(0),
    dropconnect(0),
    validationRatio(0.2),
    testRatio(0.2),
    loss(Loss::CrossEntropy),
    LRDecayConstant(0.01),
    LRStepDecay(10),
    decay(LRDecay::none),
    margin(5) // %
    {
    }

    unsigned seed;
    unsigned batchSize;
    double learningRate;
    double L1;
    double L2;
    unsigned maxEpoch;
    unsigned epochAfterOptimal;
    double dropout;
    double dropconnect;
    double validationRatio;
    double testRatio;
    Loss loss;
    double LRDecayConstant;
    unsigned LRStepDecay;
    double (* decay)(double, unsigned, double, unsigned);
    double margin; // %
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
            gradients[i][j] = (real[i][j] - predicted[i][j]);
        }
    }
    return  {loss, gradients};
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
std::pair<Matrix, Matrix> crossEntropyLoss(Matrix const& real, Matrix const& predicted)
{
    Matrix softMax = softmax(predicted);
    Matrix loss(real.size(), std::vector<double>(real[0].size(), 0));
    Matrix gradients(real.size(), std::vector<double>(real[0].size(), 0));
    for(unsigned i = 0; i < loss.size(); i++)
    {
        for(unsigned j = 0; j < loss[0].size(); j++)
        {
            loss[i][j] = real[i][j] * -std::log(softMax[i][j]);
            gradients[i][j] = real[i][j] - softMax[i][j];
        }
    }
    return  {loss, gradients};
}



} //namespace brain

#endif //BRAIN_UTILITY_HH_