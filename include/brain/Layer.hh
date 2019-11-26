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

    size_t size; //number of neurons
    double maxNorm;
    Distrib distrib;
    double mean_boundary; //mean (if normal), boundary (if uniform)
    double deviation; //deviation (if normal) or useless (if uniform)
    bool useOutput; // calculate boundary/deviation by taking output number into account
    size_t k; //number of weight set for each neuron (for maxout)
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
    virtual Matrix getGradients(ThreadPool& t) = 0;
    virtual size_t size() const = 0;
    virtual void init(size_t nbInputs, size_t nbOutputs, size_t batchSize, std::mt19937& generator) = 0;
    virtual void updateWeights(double learningRate, double L1, double L2, Optimizer opti, double momentum, double window, double optimizerBias, ThreadPool& t) = 0;
    virtual void save() = 0;
    virtual void loadSaved() = 0;
    virtual std::vector<std::pair<Matrix, Vector>> getWeights(ThreadPool& t) const = 0;
    virtual void resize(size_t neurons);
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


    void init(size_t nbInputs, size_t nbOutputs, size_t batchSize, std::mt19937& generator)
    {
        _inputSize = nbInputs;
        _batchSize = batchSize;
        try
        {
            for(size_t i = 0; i < _neurons.size(); i++)
            {
                _neurons[i].init(_param.distrib, _param.mean_boundary, _param.deviation, nbInputs, nbOutputs, batchSize, _param.k, generator, _param.useOutput);
            }
        }
        catch(std::bad_alloc const& e)
        {
            throw Exception("std::bad_alloc have been thrown while initializing the model. Try to decrease the batch size or the maxout k. Is your executable 64-bits ?");
        }
    }


    Matrix process(Matrix const& inputs, ThreadPool& t)
    {
        //lines are features, columns are neurons
        Matrix output(inputs.rows(), _neurons.size());
        std::vector<std::future<void>> tasks(_neurons.size());

        for(size_t i = 0; i < _neurons.size(); i++)
        {
            tasks[i] = t.enqueue([this, &inputs, &output, i]()->void
            {
                //one result per feature (for each neuron)
                Vector result = _neurons[i].process(inputs);
                for(eigen_size_t j = 0; j < result.size(); j++)
                    output(j, i) = result[j];
            });
        }
        for(size_t i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
        }
        return output;
    }


    Matrix processToLearn(Matrix const& inputs, double dropout, double dropconnect, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen, ThreadPool& t)
    {
        //lines are features, columns are neurons
        Matrix output(_batchSize, _neurons.size());
        std::vector<std::future<void>> tasks(_neurons.size());

        for(size_t i = 0; i < _neurons.size(); i++)
        {
            tasks[i] = t.enqueue([this, &inputs, &output, i, dropout, dropconnect, &dropoutDist, &dropconnectDist, &dropGen]()->void
            {
                //one result per feature (for each neuron)
                Vector result = _neurons[i].processToLearn(inputs, dropconnect, dropconnectDist, dropGen);
                for(eigen_size_t j = 0; j < result.size(); j++)
                {
                    output(j, i) = result[j];
                    //dropOut
                    if(dropout > std::numeric_limits<double>::epsilon())
                    {
                        if(dropoutDist(dropGen))
                            output(j, i) = 0;
                        else
                            output(j, i) /= (1-dropout);
                    }
                }
            });
        }
        for(size_t i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
        }
        return output;
    }


    void computeGradients(Matrix const& inputGradients, ThreadPool& t)
    {
        std::vector<std::future<void>> tasks(_neurons.size());

        for(size_t i = 0; i < _neurons.size(); i++)
        {
            tasks[i] = t.enqueue([this, &inputGradients, i]()->void
            {
                _neurons[i].computeGradients(inputGradients.col(i));
            });
        }
        for(size_t i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
        }
    }


    void save()
    {
        for(size_t i = 0; i < _neurons.size(); i++)
        {
            _neurons[i].save();
        }
    }


    void loadSaved()
    {
        for(size_t i = 0; i < _neurons.size(); i++)
        {
            _neurons[i].loadSaved();
        }
    }


    //one gradient per input neuron (col) and per feature (line)
    Matrix getGradients(ThreadPool& t)
    {
        Matrix grad = Matrix::Constant(_batchSize, _inputSize, 0);
        std::vector<std::future<void>> tasks(_neurons.size());

        for(size_t i = 0; i < _neurons.size(); i++)
        {
            tasks[i] = t.enqueue([this, i, &grad]()->void
            {
                Matrix neuronGrad = _neurons[i].getGradients();
                for(eigen_size_t j = 0; j < neuronGrad.rows(); j++)
                {
                    for(eigen_size_t k = 0; k < neuronGrad.cols(); k++)
                        grad(j, k) += neuronGrad(j, k);
                }
            });
        }
        for(size_t i = 0; i < tasks.size(); i++)
            tasks[i].get();
        return grad;
    }


    void updateWeights(double learningRate, double L1, double L2, Optimizer opti, double momentum, double window, double optimizerBias, ThreadPool& t)
    {
        std::vector<std::future<void>> tasks(_neurons.size());

        for(size_t i = 0; i < _neurons.size(); i++)
        {
            tasks[i] = t.enqueue([=]()->void
            {
                _neurons[i].updateWeights(learningRate, L1, L2, _param.maxNorm, opti, momentum, window, optimizerBias);
            });
        }
        for(size_t i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
        }
    }


    size_t size() const
    {
        return _neurons.size();
    }


    std::vector<std::pair<Matrix, Vector>> getWeights(ThreadPool& t) const
    {
        std::vector<std::pair<Matrix, Vector>> weights(size());
        std::vector<std::future<void>> tasks(_neurons.size());

        for(size_t i = 0; i < _neurons.size(); i++)
        {
            tasks[i] = t.enqueue([this, &weights, i]()->void
            {
                weights[i] = _neurons[i].getWeights();
            });
        }
        for(size_t i = 0; i < tasks.size(); i++)
        {
            tasks[i].get();
        }
         return weights;
    }

    void resize(size_t neurons)
    {
        _neurons = std::vector<Neuron<Aggr_t, Act_t>>(neurons);
    }

protected:

    LayerParam _param;
    size_t _inputSize;
    size_t _batchSize;
    std::vector<Neuron<Aggr_t, Act_t>> _neurons;
};



} //namespace brain



#endif //BRAIN_LAYER_HH_