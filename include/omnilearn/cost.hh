// cost.cpp

#ifndef OMNILEARN_COST_HH_
#define OMNILEARN_COST_HH_

#include "Matrix.hh"
#include "ThreadPool.hh"



namespace omnilearn
{



// one line = one feature, one colums = one class
Matrix L1Loss(Matrix const& real, Matrix const& predicted, ThreadPool& t); // use linear activation at the last layer
Vector L1Grad(Vector const& real, Vector const& predicted, ThreadPool& t);
Matrix L2Loss(Matrix const& real, Matrix const& predicted, ThreadPool& t); // use linear activation at the last layer
Vector L2Grad(Vector const& real, Vector const& predicted, ThreadPool& t);
Matrix crossEntropyLoss(Matrix const& real, Matrix const& predicted, ThreadPool& t); // use linear activation at the last layer
Vector crossEntropyGrad(Vector const& real, Vector const& predicted, ThreadPool& t);
Matrix binaryCrossEntropyLoss(Matrix const& real, Matrix const& predicted, ThreadPool& t); // use sigmoid activation at last layer (all outputs must be [0, 1])
Vector binaryCrossEntropyGrad(Vector const& real, Vector const& predicted, ThreadPool& t);



} // namespace omnilearn

#endif // OMNILEARN_COST_HH_