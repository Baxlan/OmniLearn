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


void omnilearn::Dot::init([[maybe_unused]] Distrib distrib, [[maybe_unused]] double distVal1, [[maybe_unused]] double distVal2, [[maybe_unused]] size_t nbInputs, [[maybe_unused]] size_t nbOutputs, [[maybe_unused]] std::mt19937& generator, [[maybe_unused]] bool useOutput)
{
    // nothing to do
}


double omnilearn::Dot::aggregate(Vector const& inputs, Vector const& weights) const
{
    return inputs.dot(weights);
}


omnilearn::Vector omnilearn::Dot::prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
{
    return inputs;
}


omnilearn::Vector omnilearn::Dot::primeInput([[maybe_unused]] Vector const& inputs, Vector const& weights) const
{
    return weights;
}


void omnilearn::Dot::computeGradients([[maybe_unused]] Vector const& inputs, [[maybe_unused]] Vector const& weights, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Dot::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
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


size_t omnilearn::Dot::getNbParameters() const
{
    return 0;
}


//=============================================================================
//=============================================================================
//=============================================================================
//=== DISTANCE AGGREGATION ====================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Distance::Distance(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Distance/Pdistance aggregation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _order = coefs[0];
    _savedOrder = 0;
}


void omnilearn::Distance::init([[maybe_unused]] Distrib distrib, [[maybe_unused]] double distVal1, [[maybe_unused]] double distVal2, [[maybe_unused]] size_t nbInputs, [[maybe_unused]] size_t nbOutputs, [[maybe_unused]] std::mt19937& generator, [[maybe_unused]] bool useOutput)
{
    // nothing to do
}


double omnilearn::Distance::aggregate(Vector const& inputs, Vector const& weights) const
{
    return norm((inputs.transpose() - weights).cwiseAbs(), _order);
}


omnilearn::Vector omnilearn::Distance::prime(Vector const& inputs, Vector const& weights) const
{
    double a = std::pow(aggregate(inputs, weights.transpose()), (1-_order));
    Vector result(weights.size());

    for(eigen_size_t i = 0; i < weights.size(); i++)
    {
        result[i] = (-std::pow(std::abs(inputs[i] - weights[i]), _order-1) * a);
    }
    return result;
}


omnilearn::Vector omnilearn::Distance::primeInput(Vector const& inputs, Vector const& weights) const
{
    double a = std::pow(aggregate(inputs, weights.transpose()), (1-_order));
    Vector result(weights.size());

    for(eigen_size_t i = 0; i < weights.size(); i++)
    {
        result[i] = (std::pow(std::abs(inputs[i] - weights[i]), _order-1) * a);
    }
    return result;
}


void omnilearn::Distance::computeGradients([[maybe_unused]] Vector const& inputs, [[maybe_unused]] Vector const& weights, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::Distance::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::Distance::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Distance/Pdistance aggregation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _order = coefs[0];
    _savedOrder = 0;
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


size_t omnilearn::Distance::getNbParameters() const
{
    return 0;
}


//=============================================================================
//=============================================================================
//=============================================================================
//=== PARAMETRIC DISTANCE AGGREGATION =========================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::Pdistance::Pdistance(Vector const& coefs):
Distance(coefs),
_orderInfos(),
_counter(0)
{
}


void omnilearn::Pdistance::computeGradients(Vector const& inputs, Vector const& weights, double inputGrad)
{
    Vector diff((inputs - weights).cwiseAbs());
    double calc = 0;

    for(eigen_size_t i = 0; i < inputs.size(); i++)
    {
        // log error if diff == 0. Add an epsilon ?
        calc += std::log(diff[i]) * std::pow(diff[i], _order);
    }

    _orderInfos.gradient -= inputGrad * (calc * std::pow(aggregate(inputs, weights.transpose()), (1-_order))) / _order;
    _counter += 1;
}


void omnilearn::Pdistance::updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay)
{
    _orderInfos.gradient /= static_cast<double>(_counter);

    optimizedUpdate(_order, _orderInfos, automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, decay, true);

    _orderInfos = LearnableParameterInfos();
    _counter = 0;
}


void omnilearn::Pdistance::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 1)
        throw Exception("Distance/Pdistance aggregation functions need 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
    _order = coefs[0];
    _savedOrder = 0;

    _orderInfos = LearnableParameterInfos();
    _counter = 0;
}


omnilearn::Aggregation omnilearn::Pdistance::signature() const
{
    return Aggregation::Pdistance;
}


size_t omnilearn::Pdistance::getNbParameters() const
{
    return 1;
}



//=============================================================================
//=============================================================================
//=============================================================================
//=== GRU AGGREGATION =========================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::GRU::GRU(Vector const& coefs):
_counter(0),
_cellState(0),
_sigmoid(),
_tanh()
{
    if(coefs.size() % 2 != 0)
        throw Exception("GRU aggregation function needs a pair number of coefficients. " + std::to_string(coefs.size()) + " provided.");

    _updateGateWeights = coefs.head(coefs.size()/2);
    _resetGateWeights = coefs.tail(coefs.size()/2);

    _updateGateWeightsGradient = Vector::Constant(coefs.size()/2, 0);
    _previousUpdateGateWeightsGrad = Vector::Constant(coefs.size()/2, 0);
    _previousUpdateGateWeightsGrad2 = Vector::Constant(coefs.size()/2, 0);
    _optimalPreviousUpdateGateWeightsGrad2 = Vector::Constant(coefs.size()/2, 0);
    _previousUpdateGateWeightsUpdate = Vector::Constant(coefs.size()/2, 0);

    _resetGateWeightsWeightsGradient = Vector::Constant(coefs.size()/2, 0);
    _previousResetGateWeightsGrad = Vector::Constant(coefs.size()/2, 0);
    _previousResetGateWeightsWeightsGrad2 = Vector::Constant(coefs.size()/2, 0);
    _optimalPreviousResetGateWeightsWeightsGrad2 = Vector::Constant(coefs.size()/2, 0);
    _previousResetGateWeightsWeightsUpdate = Vector::Constant(coefs.size()/2, 0);
}


void omnilearn::GRU::init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput)
{
    
}


double omnilearn::GRU::aggregate(Vector const& inputs, Vector const& weights) const
{
    //double resetGate = 
    return inputs.dot(weights);
}


omnilearn::Vector omnilearn::GRU::prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
{
    return inputs;
}


omnilearn::Vector omnilearn::GRU::primeInput([[maybe_unused]] Vector const& inputs, Vector const& weights) const
{
    return weights;
}


void omnilearn::GRU::computeGradients([[maybe_unused]] Vector const& inputs, [[maybe_unused]] Vector const& weights, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::GRU::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::GRU::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Dot aggregation function needs 0 coefficient. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::GRU::getCoefs() const
{
    return Vector(0);
}


omnilearn::Aggregation omnilearn::GRU::signature() const
{
    return Aggregation::GRU;
}


void omnilearn::GRU::keep()
{
    //nothing to do
}


void omnilearn::GRU::release()
{
    //nothing to do
}


size_t omnilearn::GRU::getNbParameters() const
{
    return 0;
}


//=============================================================================
//=============================================================================
//=============================================================================
//=== LSTM AGGREGATION ========================================================
//=============================================================================
//=============================================================================
//=============================================================================



omnilearn::LSTM::LSTM(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Dot aggregation function needs 0 coefficient. " + std::to_string(coefs.size()) + " provided.");
}


void omnilearn::LSTM::init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput)
{
    // nothing to do
}


double omnilearn::LSTM::aggregate(Vector const& inputs, Vector const& weights) const
{
    return inputs.dot(weights);
}


omnilearn::Vector omnilearn::LSTM::prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
{
    return inputs;
}


omnilearn::Vector omnilearn::LSTM::primeInput([[maybe_unused]] Vector const& inputs, Vector const& weights) const
{
    return weights;
}


void omnilearn::LSTM::computeGradients([[maybe_unused]] Vector const& inputs, [[maybe_unused]] Vector const& weights, [[maybe_unused]] double inputGrad)
{
    //nothing to do
}


void omnilearn::LSTM::updateCoefs([[maybe_unused]] bool automaticLearningRate, [[maybe_unused]] bool adaptiveLearningRate, [[maybe_unused]] bool useMaxDenominator, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum, [[maybe_unused]] double previousMomentum, [[maybe_unused]] double nextMomentum, [[maybe_unused]] double cumulativeMomentum, [[maybe_unused]] double window, [[maybe_unused]] double optimizerBias, [[maybe_unused]] size_t iteration, [[maybe_unused]] double L1, [[maybe_unused]] double L2, [[maybe_unused]] double decay)
{
    //nothing to do
}


void omnilearn::LSTM::setCoefs(Vector const& coefs)
{
    if(coefs.size() != 0)
        throw Exception("Dot aggregation function needs 0 coefficient. " + std::to_string(coefs.size()) + " provided.");
}


omnilearn::rowVector omnilearn::LSTM::getCoefs() const
{
    return Vector(0);
}


omnilearn::Aggregation omnilearn::LSTM::signature() const
{
    return Aggregation::LSTM;
}


void omnilearn::LSTM::keep()
{
    //nothing to do
}


void omnilearn::LSTM::release()
{
    //nothing to do
}


size_t omnilearn::LSTM::getNbParameters() const
{
    return 0;
}
