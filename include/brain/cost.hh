#ifndef BRAIN_COST_HH_
#define BRAIN_COST_HH_

#include <cmath>

#include "Activation.hh"
#include "matrix.hh"

namespace brain
{



// one line = one feature, one colums = one class
// first are loss, second are gradients
// use linear activation at the last layer
std::pair<Matrix, Matrix> L1Loss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix loss = Matrix::Constant(real.rows(), real.cols(), 0);
    Matrix gradients = Matrix::Constant(real.rows(), real.cols(), 0);
    std::vector<std::future<void>> tasks;
    for(size_t i = 0; i < loss.rows(); i++)
    {
        tasks.push_back(t.enqueue([&real, &predicted, &loss, &gradients, i]()->void
        {
            for(size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = std::abs(real(i, j) - predicted(i, j));
                if (real(i, j) < predicted(i, j))
                    gradients(i, j) = -1;
                else if (real(i, j) > predicted(i, j))
                    gradients(i, j) = 1;
                else
                    gradients(i, j) = 0;
            }
        }));
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return {loss, gradients};
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
// use linear activation at the last layer
std::pair<Matrix, Matrix> L2Loss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix loss = Matrix::Constant(real.rows(), real.cols(), 0);
    Matrix gradients = Matrix::Constant(real.rows(), real.cols(), 0);
    std::vector<std::future<void>> tasks;
    for(size_t i = 0; i < loss.rows(); i++)
    {
        tasks.push_back(t.enqueue([&real, &predicted, &loss, &gradients, i]()->void
        {
            for(size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = 0.5 * std::pow(real(i, j) - predicted(i, j), 2);
                gradients(i, j) = (real(i, j) - predicted(i, j));
            }
        }));
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return  {loss, gradients};
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
// use linear activation at the last layer
std::pair<Matrix, Matrix> crossEntropyLoss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix softMax = softmax(predicted);
    Matrix loss(real.rows(), real.cols());
    Matrix gradients(real.rows(), real.cols());
    std::vector<std::future<void>> tasks;
    for(size_t i = 0; i < loss.rows(); i++)
    {
        tasks.push_back(t.enqueue([&real, &predicted, &softMax, &loss, &gradients, i]()->void
        {
            for(size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = real(i, j) * -std::log(softMax(i, j));
                gradients(i, j) = real(i, j) - softMax(i, j);
            }
        }));
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return  {loss, gradients};
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
// use sigmoid activation at last layer (all outputs must be [0, 1])
std::pair<Matrix, Matrix> binaryCrossEntropyLoss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix loss = Matrix::Constant(real.rows(), real.cols(), 0);
    Matrix gradients = Matrix::Constant(real.rows(), real.cols(), 0);
    std::vector<std::future<void>> tasks;
    for(size_t i = 0; i < loss.rows(); i++)
    {
        tasks.push_back(t.enqueue([&real, &predicted, &loss, &gradients, i]()->void
        {
            for(size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = -(real(i, j) * std::log(predicted(i, j)) + (1 - real(i, j)) * std::log(1 - predicted(i, j)));
                gradients(i, j) = (real(i, j) - predicted(i, j)) / ( predicted(i, j) * (1 -  predicted(i, j)));
            }
        }));
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return  {loss, gradients};
}


} // namespace brain

#endif // BRAIN_COST_HH_