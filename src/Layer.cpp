// Layer.cpp

#include "omnilearn/Layer.hh"

omnilearn::LayerParam omnilearn::Layer::generateLinearLayerParam()
{
    LayerParam param = LayerParam();
    param.distrib = Distrib::Uniform;
    param.deviation_boundary = 1;
    param.useOutput = false;

    return param;
}


omnilearn::LayerParam omnilearn::Layer::generateLinearNormalizedLayerParam()
{
    LayerParam param = LayerParam();
    param.distrib = Distrib::Uniform;
    param.deviation_boundary = 6;
    param.useOutput = true;

    return param;
}


omnilearn::LayerParam omnilearn::Layer::generateNonLinearLayerParam()
{
    LayerParam param = LayerParam();
    param.distrib = Distrib::Normal;
    param.deviation_boundary = 2;
    param.useOutput = false;

    return param;
}


omnilearn::Layer::Layer(LayerParam const& param):
_param(param),
_inputSize(0),
_neurons(std::vector<Neuron>(param.size))
{
    for(size_t i = 0; i < _neurons.size(); i++)
        _neurons[i] = Neuron(param.aggregation, param.activation);
}


void omnilearn::Layer::init(size_t nbInputs, std::mt19937& generator)
{
    _inputSize = nbInputs;
    for(size_t i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].init(_param.distrib, _param.mean, _param.deviation_boundary, nbInputs, _neurons.size(), _param.k, generator, _param.useOutput);
    }
}


void omnilearn::Layer::init(size_t nbInputs)
{
    _inputSize = nbInputs;
}


omnilearn::Matrix omnilearn::Layer::process(Matrix const& inputs, ThreadPool& t) const
{
    //lines are features, columns are neurons
    Matrix output(inputs.rows(), _neurons.size());
    std::vector<std::future<void>> tasks(_neurons.size());

    // only one instance of exception_ptr is required because all threads would throw the same exception
    std::exception_ptr ep = nullptr;

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, &inputs, &output, i, &ep]()->void
        {
            try
            {
                //one result per feature (for each neuron)
                Vector result = _neurons[i].process(inputs);
                for(eigen_size_t j = 0; j < result.size(); j++)
                    output(j, i) = result[j];
            }
            catch(...)
            {
                ep = std::current_exception();
            }
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    if(ep)
        std::rethrow_exception(ep);
    return output;
}


omnilearn::Vector omnilearn::Layer::processToLearn(Vector const& input, double dropout, double dropconnect, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen, ThreadPool& t)
{
    //each element is associated to a neuron
    Vector output(_neurons.size());
    std::vector<std::future<void>> tasks(_neurons.size());

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, &input, &output, i, dropout, dropconnect, &dropoutDist, &dropconnectDist, &dropGen]()->void
        {
            output(i) = _neurons[i].processToLearn(input, dropconnect, dropconnectDist, dropGen);
            //dropOut
            if(dropout > std::numeric_limits<double>::epsilon())
            {
                if(dropoutDist(dropGen))
                    output[i] = 0;
                else
                    output[i] /= (1-dropout);
            }
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return output;
}


omnilearn::Vector omnilearn::Layer::processToGenerate(Vector const& input, ThreadPool& t)
{
    //each element is associated to a neuron
    Vector output(_neurons.size());
    std::vector<std::future<void>> tasks(_neurons.size());

    // only one instance of exception_ptr is required because all threads would throw the same exception
    std::exception_ptr ep = nullptr;

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, &input, &output, i, &ep]()->void
        {
            try
            {
                output(i) = _neurons[i].processToGenerate(input);
            }
            catch(...)
            {
                ep = std::current_exception();
            }
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    if(ep)
        std::rethrow_exception(ep);
    return output;
}


void omnilearn::Layer::computeGradients(Vector const& inputGradient, ThreadPool& t)
{
    std::vector<std::future<void>> tasks(_neurons.size());

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, &inputGradient, i]()->void
        {
            _neurons[i].computeGradients(inputGradient[i]);
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
}


void omnilearn::Layer::computeGradientsAccordingToInputs(Vector const& inputGradient, ThreadPool& t)
{
    std::vector<std::future<void>> tasks(_neurons.size());

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, &inputGradient, i]()->void
        {
            _neurons[i].computeGradientsAccordingToInputs(inputGradient[i]);
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
}


void omnilearn::Layer::keep()
{
    for(size_t i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].keep();
    }
}


void omnilearn::Layer::release()
{
    for(size_t i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].release();
    }
}


//one gradient per input neuron
omnilearn::Vector omnilearn::Layer::getGradients(ThreadPool& t)
{
    Vector grad = Vector::Constant(_inputSize, 0);
    std::vector<std::future<void>> tasks(_neurons.size());

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, i, &grad]()->void
        {
            Vector neuronGrad = _neurons[i].getGradients();
            for(eigen_size_t j = 0; j < neuronGrad.size(); j++)
                grad(j) += neuronGrad(j);
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return grad;
}


void omnilearn::Layer::updateWeights(double learningRate, double L1, double L2, double weightDecay, bool automaticLearningRate, bool adaptiveLearningRate, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, ThreadPool& t)
{
    std::vector<std::future<void>> tasks(_neurons.size());

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([=]()->void
        {
            _neurons[i].updateWeights(learningRate, L1, L2, weightDecay, _param.maxNorm, automaticLearningRate, adaptiveLearningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, _param.lockWeights, _param.lockBias);
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
}


void omnilearn::Layer::updateInput(Vector& input, double learningRate)
{
    // no parallelization here because editing the same input by multiple neurons at the same time would cause data races
    for(size_t i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].updateInput(input, learningRate);
    }
}


void omnilearn::Layer::resetGradientsForGeneration(ThreadPool& t)
{
    std::vector<std::future<void>> tasks(_neurons.size());

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, i]()->void
        {
            _neurons[i].resetGradientsForGeneration();
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
}


size_t omnilearn::Layer::size() const
{
    return _neurons.size();
}


void omnilearn::Layer::resize(size_t neurons)
{
    _neurons = std::vector<Neuron>(neurons);
    for(size_t i = 0; i < _neurons.size(); i++)
        _neurons[i] = Neuron(_param.aggregation, _param.activation);
}


size_t omnilearn::Layer::nbWeights() const
{
    return _neurons[0].nbWeights();
}


std::pair<double, double> omnilearn::Layer::L1L2(ThreadPool& t) const
{
    std::vector<std::future<void>> tasks(_neurons.size());
    std::pair<double, double> L1L2;
    double L1 = 0;
    double L2 = 0;

    for(size_t i = 0; i < _neurons.size(); i++)
    {
        tasks[i] = t.enqueue([this, &L1, &L2, &L1L2, i]()->void
        {
            L1L2 = _neurons[i].L1L2();
            L1 += L1L2.first;
            L2 += L1L2.second;
        });
    }
    for(size_t i = 0; i < tasks.size(); i++)
        tasks[i].get();
    return {L1, L2};
}


void omnilearn::to_json(json& jObj, Layer const& layer)
{
    jObj["aggregation"] = aggregationToStringMap[layer._param.aggregation];
    jObj["activation"] = activationToStringMap[layer._param.activation];
    jObj["maxnorm"] = layer._param.maxNorm;
    for(size_t i = 0; i < layer._neurons.size(); i++)
    {
        jObj["neurons"][i] = layer._neurons[i];
    }
}


void omnilearn::from_json(json const& jObj, Layer& layer)
{
    layer._param.aggregation = stringToAggregationMap[jObj.at("aggregation")];
    layer._param.activation = stringToActivationMap[jObj.at("activation")];
    layer._param.maxNorm = jObj.at("maxnorm");
    layer._neurons.resize(jObj.at("neurons").size());

    for(size_t i = 0; i < layer._neurons.size(); i++)
    {
        layer._neurons[i] = jObj.at("neurons").at(i);
    }
}