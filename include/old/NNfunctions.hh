//NNfunctions.hh

#ifndef NNFUNCTIONS_HH_
#define NNFUNCTIONS_HH_

#include <cmath>
#include <vector>
#include <algorithm>



namespace stb
{



namespace learn
{



struct ActivationProperties
{
    static double leakyReluCoef;
    static double SoftplusCoef;
    static double eluCoef;
    static double distOrder;
};
double ActivationProperties::leakyReluCoef = 0.01;
double ActivationProperties::SoftplusCoef = 0.01;
double ActivationProperties::eluCoef = 1;
double ActivationProperties::distOrder = 3;



//=============================================================================
//=============================================================================
//=============================================================================
//=== ACTIVATION FUNCTIONS =====================================================
//=============================================================================
//=============================================================================
//=============================================================================


double sigmoid(double val)
{
    return 1 / (1 + std::exp(-val));
}


double sigmoidPrime(double val)
{
    return val * (1 - val);
}


double tanh(double val)
{
    return std::tanh(val);
}


double tanhPrime(double val)
{
    return -1/static_cast<double>(std::pow(std::cosh(val),2));
}


double linear(double val)
{
    return val;
}


double linearPrime(double val)
{
    val += 0;
    return 1.;
}


//rectified linear unit
double relu(double val)
{
    return (val < 0 ? 0 : val);
}


double reluPrime(double val)
{
    return (val < 0 ? 0 : 1);
}


// leaky rectified linear unit
double leakyRelu(double val)
{
    return (val < 0 ? ActivationProperties::leakyReluCoef*val : val);
}


double leakyReluPrime(double val)
{
    return (val < 0 ? ActivationProperties::leakyReluCoef : 1);
}


// exponential linear unit
double elu(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*(std::exp(val)-1) : val);
}


double eluPrime(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*std::exp(val) : 1);
}


// Softplus is a smooth version of relu
double softplus(double val)
{
    return std::log(std::exp(val) + 1);
}


double softplusPrime(double val)
{
    return sigmoid(val);
}


//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC ACTIVATION FUNCTIONS ==========================================
//=============================================================================
//=============================================================================
//=============================================================================


/*

// parametric rectified linear unit (like leakyRelu but the coef is learnable)
double pRelu(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*(std::exp(val)-1) : val);
}


double pReluPrime(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*std::exp(val) : 1);
}


// parametric exponential linear unit (like elu but the coef is learnable)
double pElu(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*(std::exp(val)-1) : val);
}


double pEluPrime(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*std::exp(val) : 1);
}


// S-shaped rectified linear unit (combination of 3 linear functions, with 4 learnable parameters )
double sRelu(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*(std::exp(val)-1) : val);
}


double sReluPrime(double val)
{
    return (val < 0 ? ActivationProperties::eluCoef*std::exp(val) : 1);
}


*/



//=============================================================================
//=============================================================================
//=============================================================================
//=== AGREGATION FUNCTIONS =====================================================
//=============================================================================
//=============================================================================
//=============================================================================


double dot(std::vector<double> const& inputs, std::vector<double> const& weights)
{
    double val = 0;
    for(unsigned i=0; i < inputs.size(); i++)
    {
        val += inputs[i]*weights[i];
    }
    return val;
}


double dotPrime(std::vector<double> const& inputs, std::vector<double> const& weights, unsigned index)
{
    weights[0];
    return inputs[index];
}


double L1_dist(std::vector<double> const& inputs, std::vector<double> const& weights)
{
    double result;
    for(unsigned i=0; i<inputs.size(); i++)
    {
        result += (inputs[i] - weights[i]);
    }
    return result;
}

double L2_dist(std::vector<double> const& inputs, std::vector<double> const& weights)
{
    double result;
    for(unsigned i=0; i<inputs.size(); i++)
    {
        result += static_cast<double>(std::pow((inputs[i] - weights[i]), 2));
    }
    return std::pow(result, 1/2);
}


double Lp_dist(std::vector<double> const& inputs, std::vector<double> const& weights)
{
    double result;
    for(unsigned i=0; i<inputs.size(); i++)
    {
        result += static_cast<double>(std::pow((inputs[i] - weights[i]), ActivationProperties::distOrder));
    }
    return std::pow(result, 1/ActivationProperties::distOrder);
}


double Linf_dist(std::vector<double> const& inputs, std::vector<double> const& weights)
{
    std::vector<double> result(inputs.size(), 0);
    for(unsigned i=0; i<inputs.size(); i++)
    {
        result[i] = (inputs[i] - weights[i]);
    }
    return *std::max_element(result.begin(), result.end());
}

//=============================================================================
//=============================================================================
//=============================================================================
//=== COST FUNCTIONS ===========================================================
//=============================================================================
//=============================================================================
//=============================================================================


std::vector<double> cost(std::vector<double> const& real, std::vector<double> const& neural, double margin)
{
    std::vector<double> loss(real.size(), 0);
    for(unsigned i=0; i<loss.size(); i++)
    {
        loss[i] = real[i] - neural[i];
        loss[i] = std::abs(loss[i]) < margin ? 0 : loss[i];
    }
    return loss;
}


std::vector<double> cost_square(std::vector<double> const& real, std::vector<double> const& neural, double margin)
{
    std::vector<double> loss(real.size(), 0);
    for(unsigned i=0; i<loss.size(); i++)
    {
        loss[i] = static_cast<double>(std::pow(real[i] - neural[i], 2))/2;
        loss[i] = std::abs(loss[i]) < margin ? 0 : loss[i];
    }
    return loss;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== UTILITY FUNCTIONS =======================================================
//=============================================================================
//=============================================================================
//=============================================================================


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


} // namespace learn



} // namespace stb



#endif // NNFUNCTIONS_HH_