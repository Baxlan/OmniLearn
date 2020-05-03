// Aggregation.cpp

#include "omnilearn/Aggregation.hh"

#include "omnilearn/Exception.hh"



//=============================================================================
//=============================================================================
//=============================================================================
//=== DOT AGGREGATION =========================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Dot::Dot(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Dot aggregation function needs 0 coefficient. " + std::to_string(coefs.size()) + " provided.");
}


std::pair<double, size_t> omnilearn::Dot::aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
{
    if(weights.rows() > 1)
        throw Exception("Dot aggregation only requires one weight set. " + std::to_string(weights.rows()) + " provided.");
    return {inputs.dot(weights.row(0)) + bias[0], 0};
}


omnilearn::Vector omnilearn::Dot::prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
{
    return inputs;
}


omnilearn::Vector omnilearn::Dot::primeInput([[maybe_unused]] Vector const& inputs, Vector const& weights) const
{
    return weights;
}


void omnilearn::Dot::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Dot::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Dot aggregation function needs 0 coefficient. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Dot::getCoefs() const
{
    return Vector(0);
}


omnilearn::Aggregation omnilearn::Dot::signature() const
{
    return Aggregation::Dot;
}


void omnilearn::Dot::keep()
{
    //nothing to do
}


void omnilearn::Dot::release()
{
    //nothing to do
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== DISTANCE AGGREGATION ====================================================
//=============================================================================
//=============================================================================
//=============================================================================



const omnilearn::Vector omnilearn::Distance::_bias = (Vector(1) << 0).finished();


omnilearn::Distance::Distance(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Distance aggregation function needs 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _order = coefs[0];
}


std::pair<double, size_t> omnilearn::Distance::aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
{
    if(weights.rows() > 1)
        throw Exception("Distance aggregation only requires one weight set. " + std::to_string(weights.rows()) + " provided.");
    return {norm(inputs.transpose() - weights.row(0), _order) + bias[0], 0};
}


omnilearn::Vector omnilearn::Distance::prime(Vector const& inputs, Vector const& weights) const
{
    double a = std::pow(aggregate(inputs, weights.transpose(), _bias).first, (1-_order));
    Vector result(weights.size());

    for(eigen_size_t i = 0; i < weights.size(); i++)
    {
        result[i] = (-std::pow((inputs[i] - weights[i]), _order-1) * a);
    }
    return result;
}


omnilearn::Vector omnilearn::Distance::primeInput(Vector const& inputs, Vector const& weights) const
{
    double a = std::pow(aggregate(inputs, weights, _bias).first, (1-_order));
    Vector result(weights.size());

    for(eigen_size_t i = 0; i < weights.size(); i++)
    {
        result[i] = (std::pow((inputs[i] - weights[i]), _order-1) * a);
    }
    return result;
}


void omnilearn::Distance::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Distance::setCoefs([[maybe_unused]] Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Distance aggregation function needs 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _order = coefs[0];
}


omnilearn::rowVector omnilearn::Distance::getCoefs() const
{
    return (Vector(1) << _order).finished();
}


omnilearn::Aggregation omnilearn::Distance::signature() const
{
    return Aggregation::Distance;
}


void omnilearn::Distance::keep()
{
    _savedOrder = _order;
}


void omnilearn::Distance::release()
{
    _order = _savedOrder;
}

//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC DISTANCE AGGREGATION =========================================
//=============================================================================
//=============================================================================
//=============================================================================


omnilearn::Pdistance::Pdistance(Vector const& coefs):
Distance(coefs)
{
}


void omnilearn::Pdistance::learn(double gradient, double learningRate)
{

}


omnilearn::Aggregation omnilearn::Pdistance::signature() const
{
    return Aggregation::Pdistance;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== MAXOUT AGGREGATION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Maxout::Maxout(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Maxout aggregation function needs 0 coefficient. " + std::to_string(coefs.size()) + " provided.");
}


std::pair<double, size_t> omnilearn::Maxout::aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
{
    if(weights.rows() < 2)
        throw Exception("Maxout aggregation requires multiple weight sets.");

    //each index represents a weight set
    Vector dots(weights.rows());

    for(eigen_size_t i = 0; i < weights.rows(); i++)
    {
        dots[i] = inputs.dot(weights.row(i)) + bias[i];
    }

    size_t index = 0;
    double max = dots.maxCoeff(&index);
    return {max, index};
}


omnilearn::Vector omnilearn::Maxout::prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
{
    return inputs;
}


omnilearn::Vector omnilearn::Maxout::primeInput([[maybe_unused]] Vector const& inputs, Vector const& weights) const
{
    return weights;
}


void omnilearn::Maxout::learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
{
    //nothing to learn
}


void omnilearn::Maxout::setCoefs([[maybe_unused]] Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Maxout aggregation function needs 0 coefficient. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::Maxout::getCoefs() const
{
    return Vector(0);
}


omnilearn::Aggregation omnilearn::Maxout::signature() const
{
    return Aggregation::Maxout;
}


void omnilearn::Maxout::keep()
{
    //nothing to do
}


void omnilearn::Maxout::release()
{
    //nothing to do
}