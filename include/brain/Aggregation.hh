#ifndef BRAIN_AGGREGATION_HH_
#define BRAIN_AGGREGATION_HH_

#include "Matrix.hh"

namespace brain
{



class Aggregation //abstract class
{
public:
    virtual ~Aggregation(){}
    virtual std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const = 0; //double is the result, size_t is the index of the weight set used
    virtual Vector prime(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each weight (weights from the index "index")
    virtual void learn(double gradient, double learningRate) = 0;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== DOT AGGREGATION =========================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Dot : public Aggregation
{
public:
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
    {
        if(weights.lines() > 1)
            throw Exception("Dot aggregation only requires one weight set.");
        return {Vector::dot(inputs, weights[0]) + bias[0], 0};
    }


    Vector prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
    {
        return inputs;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== DISTANCE AGGREGATION ====================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Distance : public Aggregation
{
public:
    Distance(size_t order = 2):
    _order(order)
    {
    }


    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
    {
        if(weights.lines() > 1)
            throw Exception("Distance aggregation only requires one weight set.");
        return {Vector::distance(inputs, weights[0], _order) + bias[0], 0};
    }


    Vector prime(Vector const& inputs, Vector const& weights) const
    {
        double a = std::pow(aggregate(inputs, {weights}, {0}).first, (1-_order));
        Vector result(weights.size(), 0);

        for(size_t i = 0; i < weights.size(); i++)
        {
          result[i] += (-std::pow((inputs[i] - weights[i]), _order-1) * a);
        }
        return result;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

protected:
    size_t const _order;
};



//=============================================================================
//=============================================================================
//=============================================================================
//=== MAXOUT AGGREGATION ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



class Maxout : public Aggregation
{
public:
    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
    {
        if(weights.lines() < 2)
            throw Exception("Maxout aggregation requires multiple weight sets.");

        //each index represents a weight set
        Vector dots(weights.lines(), 0);

        for(size_t i = 0; i < weights.lines(); i++)
        {
            dots[i] = Vector::dot(inputs, weights[i]) + bias[i];
        }

        //max and index of the max
        std::pair<double, size_t> max = {dots[0], 0};
        for(size_t i = 0; i < dots.size(); i++)
        {
            if(dots[i] > max.first)
            {
                max.first = dots[i];
                max.second = i;
            }
        }

        return max;
    }


    Vector prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
    {
        return inputs;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }
};



} //namespace brain



#endif //BRAIN_AGGREGATION_HH_
