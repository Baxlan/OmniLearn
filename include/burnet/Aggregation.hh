#ifndef BURNET_AGGREGATION_HH_
#define BURNET_AGGREGATION_HH_

#include "utility.hh"

namespace burnet
{



class Aggregation //abstract class
{
    friend class Neuron;
public:
    Aggregation(unsigned k):
    _k(k),
    _weights(std::vector<std::vector<double>>()),
    _bias(std::vector<double>())
    {
        if (_k == 0)
        {
            throw Exception("An aggregation structure must manage at least one wheight set.");
        }
    }

    Aggregation(std::vector<std::vector<double>> const& weights, std::vector<double> const& bias):
    _k(_weights.size()),
    _weights(weights),
    _bias(bias)
    {
        if (_k == 0)
        {
            throw Exception("An aggregation structure must handle at least one weight set.");
        }
        if (_weights.size() != _bias.size())
        {
            throw Exception("An aggregation structure must handle the same amount of weight set and bias.");
        }
    }

    virtual ~Aggregation();
    virtual std::pair<double, unsigned> aggregate(std::vector<double> const& inputs) = 0; //double is the result, unsigned is the index of the weight set used
    virtual std::vector<double> prime(std::vector<double> const& inputs, unsigned index) = 0; //return derivatives according to each weight (weights from the index "index")
    virtual void learn(double gradient, double learningRate, double momentum) = 0;

    double k() const
    {
        return _k;
    }

    std::vector<std::vector<double>> weights() const
    {
        return _weights;
    }

    std::vector<double> bias() const
    {
        return _bias;
    }

private:
    virtual std::pair<std::vector<double>&, double&> weightRef(unsigned index) final
    {
        return {_weights[index], _bias[index]};
    }

protected:
    unsigned const _k; // number of weight set
    std::vector<std::vector<double>> _weights;
    std::vector<double> _bias;
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
    Dot(): Aggregation(1)
    {
    }

    Dot(std::vector<std::vector<double>> weights, std::vector<double> bias):
    Aggregation(weights, bias)
    {
        if (_k > 1)
        {
            throw Exception("The burnet::Dot aggregation structure must handle exactly one weight set.");
        }
    }

    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs)
    {
        return {dot(inputs, _weights[0]) + _bias[0], 0};
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] unsigned index)
    {
        return inputs;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
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
    Aggregation(1),
    _order(order)
    {
    }


    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs)
    {
        return {distance(inputs, _weights[0], _order) + _bias[0], 0};
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] unsigned index)
    {
        double a = std::pow(aggregate(inputs).first, (1-_order));
        std::vector<double> result(_weights.size(), 0);

        for(unsigned i = 0; i < _weights.size(); i++)
        {
          result[i] += (-std::pow((inputs[i] - _weights[0][i]), _order-1) * a);
        }
        return result;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
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
    Maxout(unsigned k):
    Aggregation(k)
    {
    }


    std::pair<double, unsigned> aggregate(std::vector<double> const& inputs)
    {
        //each index represents a weight set
        std::vector<double> dots(_k, 0);

        for(unsigned i = 0; i < _k; i++)
        {
            dots[i] = dot(inputs, _weights[i]) + _bias[i];
        }

        //max and index of the max
        std::pair<double, unsigned> max = {dots[0], 0};
        for(unsigned i = 0; i < _k; i++)
        {
            if(dots[i] > max.first)
            {
                max.first = dots[i];
                max.second = i;
            }
        }

        return max;
    }


    std::vector<double> prime(std::vector<double> const& inputs, [[maybe_unused]] unsigned index)
    {
        return inputs;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate, [[maybe_unused]] double momentum)
    {
        //nothing to learn
    }
};



} //namespace burnet



#endif //BURNET_AGGREGATION_HH_