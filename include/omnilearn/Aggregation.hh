// Aggregation.hh

#ifndef OMNILEARN_AGGREGATION_HH_
#define OMNILEARN_AGGREGATION_HH_

#include <random>

#include "Activation.hh"



namespace omnilearn
{



enum class Aggregation {Dot, Distance, Pdistance, GRU, LSTM};



// interface
class IAggregation
{
public:
    virtual ~IAggregation(){}
    virtual double aggregate(Vector const& inputs, Vector const& weights) const = 0; //double is the result, size_t is the index of the weight set used
    virtual void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput) = 0;
    virtual Vector prime(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each weight (weights from the index "index")
    virtual Vector primeInput(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each input
    virtual void computeGradients(Vector const& inputs, Vector const& weights, double inputGrad) = 0;
    virtual void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay) = 0;
    virtual void setCoefs(Vector const& coefs) = 0;
    virtual rowVector getCoefs() const = 0;
    virtual Aggregation signature() const = 0;
    virtual void keep() = 0;
    virtual void release() = 0;
    virtual size_t getNbParameters() const = 0;
};



class Dot : public IAggregation
{
public:
    Dot(Vector const& coefs = Vector(0));
    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput);
    double aggregate(Vector const& inputs, Vector const& weights) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void computeGradients(Vector const& inputs, Vector const& Weights, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Aggregation signature() const;
    void keep();
    void release();
    size_t getNbParameters() const;
};



class Distance : public IAggregation
{
public:
    Distance(Vector const& coefs = (Vector(1) << 2).finished());
    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput);
    double aggregate(Vector const& inputs, Vector const& weights) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void computeGradients(Vector const& inputs, Vector const& Weights, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Aggregation signature() const;
    void keep();
    void release();
    size_t getNbParameters() const;

protected:
    double _order;
    double _savedOrder;
};



class Pdistance : public Distance
{
public:
    Pdistance(Vector const& coefs = (Vector(1) << 2).finished());
    void computeGradients(Vector const& inputs, Vector const& Weights, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    Aggregation signature() const;
    size_t getNbParameters() const;

protected:
    LearnableParameterInfos _orderInfos;
    size_t _counter;
};



class GRU : public IAggregation
{
public:
    GRU(Vector const& coefs = Vector(0));
    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput);
    double aggregate(Vector const& inputs, Vector const& weights) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void computeGradients(Vector const& inputs, Vector const& Weights, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Aggregation signature() const;
    void keep();
    void release();
    size_t getNbParameters() const;

protected:
    Vector _updateGateWeights;
    Vector _resetGateWeights;

    Vector _updateGateWeightsGradient;
    Vector _previousUpdateGateWeightsGrad;
    Vector _previousUpdateGateWeightsGrad2;
    Vector _optimalPreviousUpdateGateWeightsGrad2;
    Vector _previousUpdateGateWeightsUpdate;

    Vector _resetGateWeightsWeightsGradient;
    Vector _previousResetGateWeightsGrad;
    Vector _previousResetGateWeightsWeightsGrad2;
    Vector _optimalPreviousResetGateWeightsWeightsGrad2;
    Vector _previousResetGateWeightsWeightsUpdate;

    size_t _counter;
    double _cellState;

    Sigmoid _sigmoid;
    Tanh _tanh;
};



class LSTM : public IAggregation
{
public:
    LSTM(Vector const& coefs = Vector(0));
    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput);
    double aggregate(Vector const& inputs, Vector const& weights) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void computeGradients(Vector const& inputs, Vector const& Weights, double inputGrad);
    void updateCoefs(bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double learningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, double L1, double L2, double decay);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Aggregation signature() const;
    void keep();
    void release();
    size_t getNbParameters() const;
};


static std::map<Aggregation, std::function<std::unique_ptr<IAggregation>()>> aggregationMap =
{
    {Aggregation::Dot, []{return std::make_unique<Dot>();}},
    {Aggregation::Distance, []{return std::make_unique<Distance>();}},
    {Aggregation::Pdistance, []{return std::make_unique<Pdistance>();}},
    {Aggregation::GRU, []{return std::make_unique<GRU>();}},
    {Aggregation::LSTM, []{return std::make_unique<LSTM>();}}
};



static std::map<Aggregation, std::function<std::unique_ptr<IAggregation>(IAggregation const&)>> copyAggregationMap =
{
    {Aggregation::Dot, [](IAggregation const& a){return std::make_unique<Dot>(static_cast<Dot const&>(a));}},
    {Aggregation::Distance, [](IAggregation const& a){return std::make_unique<Distance>(static_cast<Distance const&>(a));}},
    {Aggregation::Pdistance, [](IAggregation const& a){return std::make_unique<Pdistance>(static_cast<Pdistance const&>(a));}},
    {Aggregation::GRU, [](IAggregation const& a){return std::make_unique<GRU>(static_cast<GRU const&>(a));}},
    {Aggregation::LSTM, [](IAggregation const& a){return std::make_unique<LSTM>(static_cast<LSTM const&>(a));}}
};



static std::map<std::string, Aggregation> stringToAggregationMap =
{
    {"dot", Aggregation::Dot},
    {"distance", Aggregation::Distance},
    {"pdistance", Aggregation::Pdistance},
    {"GRU", Aggregation::GRU},
    {"LSTM", Aggregation::LSTM}
};



static std::map<Aggregation, std::string> aggregationToStringMap =
{
    {Aggregation::Dot, "dot"},
    {Aggregation::Distance, "distance"},
    {Aggregation::Pdistance, "pdistance"},
    {Aggregation::GRU, "GRU"},
    {Aggregation::LSTM, "LSTM"}
};



} //namespace omnilearn



#endif //OMNILEARN_AGGREGATION_HH_
