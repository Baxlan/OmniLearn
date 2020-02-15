#ifndef BRAIN_AGGREGATION_HH_
#define BRAIN_AGGREGATION_HH_

#include "Exception.hh"
#include "Matrix.hh"

namespace brain
{



class Aggregation //abstract class
{
public:
    virtual ~Aggregation(){}
    virtual std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const = 0; //double is the result, size_t is the index of the weight set used
    virtual Vector prime(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each weight (weights from the index "index")
    virtual Vector primeInput(Vector const& inputs, Vector const& weights) const = 0; //return derivatives according to each input
    virtual void learn(double gradient, double learningRate) = 0;
    virtual void setCoefs(Vector const& coefs) = 0;
    virtual rowVector getCoefs() const = 0;
};

/*
* id 0  : dot
* id 1  : distance
* id 2  : maxout
*/

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
        if(weights.rows() > 1)
            throw Exception("Dot aggregation only requires one weight set.");
        return {inputs.dot(weights.row(0)) + bias[0], 0};
    }


    Vector prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
    {
        return inputs;
    }


    Vector primeInput(Vector const& inputs, Vector const& weights) const
    {
        return weights;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        //nothing to do
    }

    rowVector getCoefs() const
    {
        return Vector(0);
    }

    static size_t id()
    {
        return 0;
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
    Distance(Vector const& coefs = (Vector(1) << 2).finished())
    {
        if(coefs.size() != 1)
            throw Exception("Distance aggregation function needs 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
        _order = coefs[0];
    }


    std::pair<double, size_t> aggregate(Vector const& inputs, Matrix const& weights, Vector const& bias) const
    {
        if(weights.rows() > 1)
            throw Exception("Distance aggregation only requires one weight set.");
        return {norm(inputs.transpose() - weights.row(0), _order) + bias[0], 0};
    }


    Vector prime(Vector const& inputs, Vector const& weights) const
    {
        double a = std::pow(aggregate(inputs, weights.transpose(), _bias).first, (1-_order));
        Vector result(weights.size());

        for(eigen_size_t i = 0; i < weights.size(); i++)
        {
          result[i] = (-std::pow((inputs[i] - weights[i]), _order-1) * a);
        }
        return result;
    }


    //MAYBE WRONG, TO INVESTIGATE
    Vector primeInput(Vector const& inputs, Vector const& weights) const
    {
        double a = std::pow(aggregate(inputs, weights, _bias).first, (1-_order));
        Vector result(weights.size());

        for(eigen_size_t i = 0; i < weights.size(); i++)
        {
          result[i] = (-std::pow((inputs[i] - weights[i]), _order-1) * a);
        }
        return result;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        if(coefs.size() != 1)
            throw Exception("Distance aggregation function needs 1 coefficient. " + std::to_string(coefs.size()) + " provided.");
        _order = coefs[0];
    }

    rowVector getCoefs() const
    {
        return (Vector(1) << _order).finished();
    }

    static size_t id()
    {
        return 1;
    }

protected:
    size_t _order;
    static const Vector _bias;
};

const Vector Distance::_bias = (Vector(1) << 0).finished();

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


    Vector prime(Vector const& inputs, [[maybe_unused]] Vector const& weights) const
    {
        return inputs;
    }


    Vector primeInput(Vector const& inputs, Vector const& weights) const
    {
        return weights;
    }


    void learn([[maybe_unused]] double gradient, [[maybe_unused]] double learningRate)
    {
        //nothing to learn
    }

    void setCoefs([[maybe_unused]] Vector const& coefs)
    {
        //nothing to do
    }

    rowVector getCoefs() const
    {
        return Vector(0);
    }

    static size_t id()
    {
        return 2;
    }
};



} //namespace brain



#endif //BRAIN_AGGREGATION_HH_
