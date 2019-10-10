#ifndef BRAIN_LAYER_HH_
#define BRAIN_LAYER_HH_

#include "Neuron.hh"
#include "ThreadPool.hh"

namespace brain
{



//=============================================================================
//=============================================================================
//=============================================================================
//=== LAYER PARAMETERS ========================================================
//=============================================================================
//=============================================================================
//=============================================================================


struct LayerParam
{
    LayerParam():
    size(8),
    maxNorm(32),
    distrib(Distrib::Normal),
    mean_boundary(distrib == Distrib::Normal ? 0 : 6),
    deviation(2),
    useOutput(true),
    k(1)
    {
    }

    unsigned size; //number of neurons
    double maxNorm;
    Distrib distrib;
    double mean_boundary; //mean (if normal), boundary (if uniform)
    double deviation; //deviation (if normal) or useless (if uniform)
    bool useOutput; // calculate boundary/deviation by taking output number into account
    unsigned k; //number of weight set for each neuron (for maxout)
};


//=============================================================================
//=============================================================================
//=============================================================================
//=== ILAYER ==================================================================
//=============================================================================
//=============================================================================
//=============================================================================


class ILayer
{
public:
    virtual ~ILayer(){}
    virtual Matrix process(Matrix const& inputs, ThreadPool& t) = 0;
    virtual Matrix processToLearn(Matrix const& inputs, double dropout, double dropconnect, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen, ThreadPool& t) = 0;
    virtual void computeGradients(Matrix const& inputGradients, ThreadPool& t) = 0;
    virtual Matrix getGradients() = 0;
    virtual unsigned size() const = 0;
    virtual void init(unsigned nbInputs, unsigned nbOutputs, unsigned batchSize, std::mt19937& generator) = 0;
    virtual void updateWeights(double learningRate, double L1, double L2, Optimizer opti, double momentum, double window, ThreadPool& t) = 0;
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
    _param(param),
    _inputSize(0),
    _batchSize(0),
    _neurons(neurons.size() == 0 ? std::vector<Neuron<Aggr_t, Act_t>>(param.size) : neurons)
    {
    }


    void init(unsigned nbInputs, unsigned nbOutputs, unsigned batchSize, std::mt19937& generator)
    {
        _inputSize = nbInputs;
        _batchSize = batchSize;
        for(unsigned i = 0; i < size(); i++)
        {
            _neurons[i].init(_param.distrib, _param.mean_boundary, _param.deviation, nbInputs, nbOutputs, batchSize, _param.k, generator, _param.useOutput);
        }
    }


    Matrix process(Matrix const& inputs, ThreadPool& t)
    {
        //lines are features, columns are neurons
        Matrix output(inputs.size(), std::vector<double>(_neurons.size(), 0));
        std::vector<std::future<void>> tasks;

        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            tasks.push_back(t.enqueue([this, &inputs, &output, i]()->void
            {
                //one result per feature (for each neuron)
                std::vector<double> result = _neurons[i].process(inputs);
                for(unsigned j = 0; j < result.size(); j++)
                    output[j][i] = result[j];
            }));
        }
        for(unsigned i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
        }
        return output;
    }


    Matrix processToLearn(Matrix const& inputs, double dropout, double dropconnect, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen, ThreadPool& t)
    {
        //lines are features, columns are neurons
        Matrix output(_batchSize, std::vector<double>(_neurons.size(), 0));
        std::vector<std::future<void>> tasks;

        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            tasks.push_back(t.enqueue([this, &inputs, &output, i, dropout, dropconnect, &dropoutDist, &dropconnectDist, &dropGen]()->void
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
            }));
        }
        for(unsigned i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
        }
        return output;
    }


    void computeGradients(Matrix const& inputGradients, ThreadPool& t)
    {
        std::vector<std::future<void>> tasks;

        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            tasks.push_back(t.enqueue([this, &inputGradients, i]()->void
            {
                _neurons[i].computeGradients(inputGradients[i]);
            }));
        }
        for(unsigned i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
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
    //SHOULD THIS FUNCTION BE PARALELLIZED ?
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


    void updateWeights(double learningRate, double L1, double L2, Optimizer opti, double momentum, double window, ThreadPool& t)
    {
        std::vector<std::future<void>> tasks;

        for(unsigned i = 0; i < _neurons.size(); i++)
        {
            tasks.push_back(t.enqueue([=]()->void
            {
                _neurons[i].updateWeights(learningRate, L1, L2, _param.maxNorm, opti, momentum, window);
            }));
        }
        for(unsigned i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
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

    LayerParam _param;
    unsigned _inputSize;
    unsigned _batchSize;
    std::vector<Neuron<Aggr_t, Act_t>> _neurons;
};



} //namespace brain



#endif //BRAIN_LAYER_HH_