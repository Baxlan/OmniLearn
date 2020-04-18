// Aggregation.cpp

#include "omnilearn/Aggregation.hh"



//=============================================================================
//=============================================================================
//=============================================================================
//=== DOT AGGREGATION =========================================================
//=============================================================================
//=============================================================================
//=============================================================================



std::pair<double, size_t> omnilearn::Dot::aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
{
    if(weights.rows() > 1)
        throw Exception("Dot aggregation only requires one weight set.");
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


void omnilearn::Dot::setCoefs([[maybe_unused]] Vector const& coefs)
{
    //nothing to do
}


omnilearn::rowVector omnilearn::Dot::getCoefs() const
{
    return Vector(0);
}


size_t omnilearn::Dot::id() const
{
    return 0;
}


void omnilearn::Dot::save()
{
    //nothing to do
}


void omnilearn::Dot::loadSaved()
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
        throw Exception("Distance aggregation only requires one weight set.");
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


size_t omnilearn::Distance::id() const
{
    return 1;
}


void omnilearn::Distance::save()
{
    _savedOrder = _order;
}


void omnilearn::Distance::loadSaved()
{
    _order = _savedOrder;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== MAXOUT AGGREGATION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



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
    //nothing to do
}


omnilearn::rowVector omnilearn::Maxout::getCoefs() const
{
    return Vector(0);
}


size_t omnilearn::Maxout::id() const
{
    return 2;
}


void omnilearn::Maxout::save()
{
    //nothing to do
}


void omnilearn::Maxout::loadSaved()
{
    //nothing to do
}