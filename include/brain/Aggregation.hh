#ifndef BRAIN_AGGREGATION_HH_
#define BRAIN_AGGREGATION_HH_

#include "utility.hh"

namespace brain
{



class Aggregation //abstract class
{
public:
    virtual ~Aggregation(){}
    virtual std::pair<double, unsigned> aggregate(std::vector<double> const& inputs, Matrix const& weights, std::vector<double> const& bias) const = 0; //double is the result, unsigned is the index of the weight set used
    virtual std::vector<double> prime(std::vector<double> const& inputs, std::vector<double> const& weights) const = 0; //return derivatives according to each weight (weights from the index "index")
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
    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs, Matrix const& weights, std::vector<double> const& bias) const
    {
        if(weights.size() > 1)
            throw Exception("Dot aggregation only requires one weight set.");
        return {dot(inputs, weights[0]) + bias[0], 0};
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] std::vector<double> const& weights) const
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
    Distance(unsigned order = 2):
    _order(order)
    {
    }


    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs, Matrix const& weights, std::vector<double> const& bias) const
    {
        if(weights.size() > 1)
            throw Exception("Distance aggregation only requires one weight set.");
        return {distance(inputs, weights[0], _order) + bias[0], 0};
    }


    std::vector<double> prime(std::vector<double> const& inputs, std::vector<double> const& weights) const
    {
        double a = std::pow(aggregate(inputs, {weights}, {0}).first, (1-_order));
        std::vector<double> result(weights.size(), 0);

        for(unsigned i = 0; i < weights.size(); i++)
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
    unsigned const _order;
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
    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs, Matrix const& weights, std::vector<double> const& bias) const
    {
        if(weights.size() < 2)
            throw Exception("Maxout aggregation requires multiple weight sets.");

        //each index represents a weight set
        std::vector<double> dots(weights.size(), 0);

        for(unsigned i = 0; i < weights.size(); i++)
        {
            dots[i] = dot(inputs, weights[i]) + bias[i];
        }

        //max and index of the max
        std::pair<double, unsigned> max = {dots[0], 0};
        for(unsigned i = 0; i < dots.size(); i++)
        {
            if(dots[i] > max.first)
            {
                max.first = dots[i];
                max.second = i;
            }
        }

        return max;
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] std::vector<double> const& weights) const
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
