#ifndef BRAIN_COST_HH_
#define BRAIN_COST_HH_

#include "Activation.hh"

namespace brain
{



// one line = one feature, one colums = one class
// first are loss, second are gradients
// use linear activation at the last layer
Matrix L1Loss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix loss(real.rows(), real.cols());
    std::vector<std::future<void>> tasks(loss.rows());
    for(eigen_size_t i = 0; i < loss.rows(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &loss, i]()->void
        {
            for(eigen_size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = std::abs(real(i, j) - predicted(i, j));
            }
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return loss;
}


Vector L1Grad(Vector const& real, Vector const& predicted, ThreadPool& t)
{
    Vector gradients(real.size());
    std::vector<std::future<void>> tasks(real.size());
    for(eigen_size_t i = 0; i < real.size(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &gradients, i]()->void
        {
            if (real(i) < predicted(i))
                gradients(i) = -1;
            else if (real(i) > predicted(i))
                gradients(i) = 1;
            else
                gradients(i) = 0;
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return gradients;
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
// use linear activation at the last layer
Matrix L2Loss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix loss(real.rows(), real.cols());
    std::vector<std::future<void>> tasks(loss.rows());
    for(eigen_size_t i = 0; i < loss.rows(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &loss, i]()->void
        {
            for(eigen_size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = 0.5 * std::pow(real(i, j) - predicted(i, j), 2);
            }
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return loss;
}


Vector L2Grad(Vector const& real, Vector const& predicted, ThreadPool& t)
{
    Vector gradients(real.size());
    std::vector<std::future<void>> tasks(real.size());
    for(eigen_size_t i = 0; i < real.size(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &gradients, i]()->void
        {
            gradients(i) = (real(i) - predicted(i));
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return gradients;
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
// use linear activation at the last layer
Matrix crossEntropyLoss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix softMax = softmax(predicted);
    Matrix loss(real.rows(), real.cols());
    std::vector<std::future<void>> tasks(loss.rows());
    for(eigen_size_t i = 0; i < loss.rows(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &softMax, &loss, i]()->void
        {
            for(eigen_size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = real(i, j) * -std::log(softMax(i, j));
            }
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return loss;
}


Vector crossEntropyGrad(Vector const& real, Vector const& predicted, ThreadPool& t)
{
    Vector softMax = singleSoftmax(predicted);
    Vector gradients(real.size());
    std::vector<std::future<void>> tasks(real.size());
    for(eigen_size_t i = 0; i < real.size(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &softMax, &gradients, i]()->void
        {
            gradients(i) = real(i) - softMax(i);
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return gradients;
}


// one line = one feature, one colums = one class
// first are loss, second are gradients
// use sigmoid activation at last layer (all outputs must be [0, 1])
Matrix binaryCrossEntropyLoss(Matrix const& real, Matrix const& predicted, ThreadPool& t)
{
    Matrix loss(real.rows(), real.cols());
    std::vector<std::future<void>> tasks(loss.rows());
    for(eigen_size_t i = 0; i < loss.rows(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &loss, i]()->void
        {
            for(eigen_size_t j = 0; j < loss.cols(); j++)
            {
                loss(i, j) = -(real(i, j) * std::log(predicted(i, j)) + (1 - real(i, j)) * std::log(1 - predicted(i, j)));
            }
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return loss;
}


Vector binaryCrossEntropyGrad(Vector const& real, Vector const& predicted, ThreadPool& t)
{
    Vector gradients(real.size());
    std::vector<std::future<void>> tasks(real.size());
    for(eigen_size_t i = 0; i < real.size(); i++)
    {
        tasks[i] = t.enqueue([&real, &predicted, &gradients, i]()->void
        {
            gradients(i) = (real(i) - predicted(i)) / ( predicted(i) * (1 -  predicted(i)));
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return gradients;
}


} // namespace brain

#endif // BRAIN_COST_HH_