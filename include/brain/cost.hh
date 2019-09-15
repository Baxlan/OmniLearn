#ifndef BRAIN_COST_HH_
#define BRAIN_COST_HH_

#include <cmath>

#include "Activation.hh"
#include "vectorial.hh"

namespace brain
{



// one line = one feature, one colums = one class
// first are loss, second are gradients
// use linear activation at the last layer
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
// use linear activation at the last layer
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
// use linear activation at the last layer
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


// one line = one feature, one colums = one class
// first are loss, second are gradients
// use sigmoid activation at last layer (all outputs must be [0, 1])
std::pair<Matrix, Matrix> binaryCrossEntropyLoss(Matrix const& real, Matrix const& predicted)
{
    Matrix loss(real.size(), std::vector<double>(real[0].size(), 0));
    Matrix gradients(real.size(), std::vector<double>(real[0].size(), 0));
    for(unsigned i = 0; i < loss.size(); i++)
    {
        for(unsigned j = 0; j < loss[0].size(); j++)
        {
            loss[i][j] = -(real[i][j] * std::log(predicted[i][j]) + (1 - real[i][j]) * std::log(1 - predicted[i][j]));
            gradients[i][j] = real[i][j] - predicted[i][j];
        }
    }
    return  {loss, gradients};
}


} // namespace brain

#endif // BRAIN_COST_HH_