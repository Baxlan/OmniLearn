#ifndef BURNET_LAYER_HH_
#define BURNET_LAYER_HH_

#include "Neuron.hh"
#include <iostream>

namespace burnet
{


class ILayer
{
public:
    virtual ~ILayer(){}
    virtual Matrix process(Matrix const& inputs) const = 0;
    virtual Matrix processToLearn(Matrix const& inputs, double dropout, double dropconnect, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen) = 0;
    virtual void computeGradients(Matrix const& inputGradients) = 0;
    virtual Matrix getGradients() = 0;
    virtual unsigned size() const = 0;
    virtual void init(unsigned nbInputs, unsigned nbOutputs, unsigned batchSize) = 0;
    virtual void updateWeights(double learningRate, double L1, double L2, double momentum) = 0;
    virtual void save() = 0;
    virtual void loadSaved() = 0;
    virtual std::vector<std::pair<Matrix, std::vector<double>>> getWeights() const = 0;
};


//=============================================================================
//=============================================================================
//=============================================================================
//=== LAYER ===================================================================
//=============================================================================
//=============================================================================
//=============================================================================


template<typename Aggr_t = Dot, typename Act_t = Relu,
typename = typename std::enable_if<
std::is_base_of<Activation, Act_t>::value &&
std::is_base_of<Aggregation, Aggr_t>::value,
void>::type>
class Layer : public ILayer
{
public:
    Layer(LayerParam const& param = LayerParam(), std::vector<Neuron<Aggr_t, Act_t>> const& neurons = std::vector<Neuron<Aggr_t, Act_t>>()):
    _inputSize(0),
    _batchSize(0),
    _distrib(param.distrib),
    _distVal1(param.mean_boundary),
    _distVal2(param.deviation),
    _maxNorm(param.maxNorm),
    _k(param.k),
    _neurons(neurons.size() == 0 ? std::vector<Neuron<Aggr_t, Act_t>>(param.size) : neurons)
    {
    }


    void init(unsigned nbInputs, unsigned nbOutputs, unsigned batchSize)
    {
        _inputSize = nbInputs;
        _batchSize = batchSize;
        for(unsigned i = 0; i < size(); i++)
        {
            _neurons[i].init(_distrib, _distVal1, _distVal2, nbInputs, nbOutputs, batchSize, _k);
        }
    }


    Matrix process(Matrix const& inputs) const
    {
        //lines are features, columns are neurons
        Matrix output(inputs.size(), std::vector<double>(_neurons.size(), 0));
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            //one result per feature (for each neuron)
            std::vector<double> result = _neurons[i].process(inputs);
            for(unsigned j = 0; j < result.size(); j++)
            {
                output[j][i] = result[j];
            }
        }
        return output;
    }


    Matrix processToLearn(Matrix const& inputs, double dropout, double dropconnect, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen)
    {
        //lines are features, columns are neurons
        Matrix output(_batchSize, std::vector<double>(_neurons.size(), 0));
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            //one result per feature (for each neuron)
            std::vector<double> result = _neurons[i].processToLearn(inputs, dropconnect, dropconnectDist, dropGen);
            for(unsigned j = 0; j < result.size(); j++)
            {
                output[j][i] = result[j];
                //dropOut
                if(dropout > std::numeric_limits<double>::epsilon())
                {
                    if(dropoutDist(dropGen))
                        output[j][i] = 0;
                    else
                        output[j][i] /= (1-dropout);
                }
            }
        }
        return output;
    }


    void computeGradients(Matrix const& inputGradients)
    {
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            _neurons[i].computeGradients(inputGradients[i]);
        }
    }


    void save()
    {
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            _neurons[i].save();
        }
    }


    void loadSaved()
    {
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            _neurons[i].loadSaved();
        }
    }


    //one gradient per input neuron (line) and per feature (col)
    Matrix getGradients()
    {
        Matrix grad(_inputSize, std::vector<double>(_batchSize, 0));

        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            Matrix neuronGrad = _neurons[i].getGradients();
            for(unsigned j = 0; j < neuronGrad.size(); j++)
            {
                for(unsigned k = 0; k < neuronGrad[0].size(); k++)
                grad[k][j] += neuronGrad[j][k];
            }
        }

        return grad;
    }


    void updateWeights(double learningRate, double L1, double L2, double momentum)
    {
        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            _neurons[i].updateWeights(learningRate, L1, L2, _maxNorm, momentum);
        }
    }


    unsigned size() const
    {
        return _neurons.size();
    }


    std::vector<std::pair<Matrix, std::vector<double>>> getWeights() const
    {
        std::vector<std::pair<Matrix, std::vector<double>>> weights(size());

        for(unsigned i = 0; i < size(); i++)
         {
             weights[i] = _neurons[i].getWeights();
         }

         return weights;
    }


protected:

    unsigned _inputSize;
    unsigned _batchSize;
    Distrib _distrib;
    double _distVal1; //mean (if uniform), boundary (if uniform)
    double _distVal2; //deviation (if normal) or useless (if uniform)
    double const _maxNorm;
    unsigned _k; //number of weight set for each neuron
    std::vector<Neuron<Aggr_t, Act_t>> _neurons;
};



} //namespace burnet



#endif //BURNET_LAYER_HH_