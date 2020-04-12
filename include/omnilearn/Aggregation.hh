// Aggregation.hh

#ifndef OMNILEARN_AGGREGATION_HH_
#define OMNILEARN_AGGREGATION_HH_

#include <map>
#include <memory>

#include "Exception.hh"
#include "Matrix.hh"



namespace omnilearn
{



namespace Aggregation
{
static const size_t Dot = 0;
static const size_t Distance = 1;
static const size_t Maxout = 2;
} // namespace Aggregation



// interface
class AggregationFunc
{
public:
    virtual ~AggregationFunc(){}
    virtual std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const = 0; //double is the result, size_t is the index of the weight set used
    virtual Vector prime(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each weight (weights from the index "index")
    virtual Vector primeInput(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each input
    virtual void learn(double gradient, double learningRate) = 0;
    virtual void setCoefs(Vector const& coefs) = 0;
    virtual rowVector getCoefs() const = 0;
    virtual size_t id() const = 0;
    virtual void save() = 0;
    virtual void loadSaved() = 0;
};



/*
* id 0  : dot
* id 1  : distance
* id 2  : maxout
*/



class Dot : public AggregationFunc
{
public:
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();
};



class Distance : public AggregationFunc
{
public:
    Distance(Vector const& coefs = (Vector(1) << 2).finished());
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();

protected:
    double _order;
    double _savedOrder;
    static const Vector _bias;
};



class Maxout : public AggregationFunc
{
public:
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const;
    Vector prime(Vector const& inputs, Vector const& weights) const;
    Vector primeInput(Vector const& inputs, Vector const& weights) const;
    void learn(double gradient, double learningRate);
    void setCoefs(Vector const& coefs);
    rowVector getCoefs() const;
    size_t id() const;
    void save();
    void loadSaved();
};



static std::map<size_t, std::function<std::shared_ptr<AggregationFunc>()>> aggregationMap = {
    {0, []{return std::make_shared<Dot>();}},
    {1, []{return std::make_shared<Distance>();}},
    {2, []{return std::make_shared<Maxout>();}},
};



} //namespace omnilearn



#endif //OMNILEARN_AGGREGATION_HH_
