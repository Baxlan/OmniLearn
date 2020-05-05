// Aggregation.hh

#ifndef OMNILEARN_AGGREGATION_HH_
#define OMNILEARN_AGGREGATION_HH_

#include <map>
#include <memory>
#include <string>

#include "Matrix.hh"



namespace omnilearn
{



enum class Aggregation {Dot, Distance, Pdistance, Maxout};



// interface
class IAggregation
{
public:
    virtual ~IAggregation(){}
    virtual std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const = 0; //double is the result, size_t is the index of the weight set used
    virtual Vector prime(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each weight (weights from the index "index")
    virtual Vector primeInput(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each input
    virtual void learn(double gradient, double learningRate) = 0;
    virtual void setCoefs(Vector const& coefs) = 0;
    virtual rowVector getCoefs() const = 0;
    virtual Aggregation signature() const = 0;
    virtual void keep() = 0;
    virtual void release() = 0;
};



/*
* id 0  : dot
* id 1  : distance
* id 2  : maxout
*/



class Dot : public IAggregation
{
public:
    Dot(Vector const& coefs = Vector(0));
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Aggregation signature() const;
    void keep();
    void release();
};



class Distance : public IAggregation
{
public:
    Distance(Vector const& coefs = (Vector(1) << 2).finished());
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Aggregation signature() const;
    void keep();
    void release();

protected:
    double _order;
    double _savedOrder;
    static const Vector _bias;
};



class Pdistance : public Distance
{
public:
    Pdistance(Vector const& coefs = (Vector(1) << 2).finished());
    void learn(double gradient, double learningRate);
    Aggregation signature() const;
};



class Maxout : public IAggregation
{
public:
    Maxout(Vector const& coefs = Vector(0));
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    Aggregation signature() const;
    void keep();
    void release();
};



static std::map<Aggregation, std::function<std::shared_ptr<IAggregation>()>> aggregationMap = {
    {Aggregation::Dot, []{return std::make_shared<Dot>();}},
    {Aggregation::Distance, []{return std::make_shared<Distance>();}},
    {Aggregation::Pdistance, []{return std::make_shared<Pdistance>();}},
    {Aggregation::Maxout, []{return std::make_shared<Maxout>();}}
};



static std::map<std::string, Aggregation> stringToAggregationMap = {
    {"dot", Aggregation::Dot},
    {"distance", Aggregation::Distance},
    {"pdistance", Aggregation::Pdistance},
    {"maxout", Aggregation::Maxout}
};



static std::map<Aggregation, std::string> aggregationToStringMap = {
    {Aggregation::Dot, "dot"},
    {Aggregation::Distance, "distance"},
    {Aggregation::Pdistance, "pdistance"},
    {Aggregation::Maxout, "maxout"}
};



} //namespace omnilearn



#endif //OMNILEARN_AGGREGATION_HH_
