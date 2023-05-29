// Neuron.cpp

#include "omnilearn/Neuron.hh"
#include "omnilearn/optimizer.h"



omnilearn::Neuron::Neuron(Aggregation aggregation, Activation activation):
_aggregation(aggregationMap[aggregation]()),
_activation(activationMap[activation]()),
_activationType(activation),
_aggregationType(aggregation),
_weights(Vector(0)),
_input(),
_aggregResult(),
_actResult(),
_dropped(false),
_connectDropped(),
_actGradient(),
_featureGradient(),
_weightInfos(),
_count(0),
_counts(),
_savedWeights(),
_generativeGradients()
{
}


void omnilearn::Neuron::init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput)
{
    if(_weights.size() == 0)
    {
        _weights = Vector(nbInputs+1); // +1 because last element is the bias
    }
    if(distrib == Distrib::Normal)
    {
        double deviation = std::sqrt(distVal2 / static_cast<double>(nbInputs + (useOutput ? nbOutputs : 0)));
        std::normal_distribution<double> normalDist(distVal1, deviation);
        for(eigen_size_t i = 0; i < _weights.size(); i++)
            _weights(i) = normalDist(generator);
    }
    else if(distrib == Distrib::Uniform)
    {
        double boundary = std::sqrt(distVal2 / static_cast<double>(nbInputs + (useOutput ? nbOutputs : 0)));
        std::uniform_real_distribution<double> uniformDist(-boundary, boundary);
        for(eigen_size_t i = 0; i < _weights.size(); i++)
                _weights(i) = uniformDist(generator);
    }
    _weightInfos = std::vector<LearnableParameterInfos>(_weights.size());
    _count = 0;
    _counts = Size_tVector::Constant(_weights.size(), 0);
    _generativeGradients = Vector::Constant(_weights.size()-1, 0); // -1 to avoid bias

    _aggregation->init(distrib, distVal1, distVal2, nbInputs, nbOutputs, generator, useOutput);
}


//each line of the input matrix is a feature. Returns one result per feature.
omnilearn::Vector omnilearn::Neuron::process(Matrix const& inputs) const
{
    Vector results = Vector(inputs.rows());
    for(eigen_size_t i = 0; i < inputs.rows(); i++)
        results[i] = _activation->activate(_aggregation->aggregate(inputs.row(i), _weights));
    return results;
}


double omnilearn::Neuron::processToLearn(Vector const& input, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen)
{
    _input = input;
    _dropped = dropoutDist(dropGen);
    _connectDropped = BoolVector::Constant(_weights.size(), false);

    if(_dropped)
    {
        _actResult = 0;
    }
    else
    {
        if(dropconnectDist.p() > std::numeric_limits<double>::epsilon())
        {
            for(eigen_size_t i=0; i<_input.size(); i++)
            {
                if(dropconnectDist(dropGen))
                {
                    _input[i] = 0;
                    _connectDropped[i] = true;
                }
                else
                {
                    _input[i] /= (1-dropconnectDist.p());
                }
            }
        }

         //processing
        _aggregResult = _aggregation->aggregate(_input, _weights);
        _actResult = _activation->activate(_aggregResult) / (1-dropoutDist.p());
    }

    return _actResult;
}


double omnilearn::Neuron::processToGenerate(Vector const& input)
{
    _input = input;

    //processing
    _aggregResult = _aggregation->aggregate(_input, _weights);
    _actResult = _activation->activate(_aggregResult);

    return _actResult;
}


//compute gradients for one feature, finally summed for the whole batch
void omnilearn::Neuron::computeGradients(double inputGradient)
{
    if(!_dropped)
    {
        _featureGradient = Vector(_weights.size()-1);

        _actGradient = _activation->prime(_aggregResult);
        Vector grad(_aggregation->prime(_input, _weights));

        _activation->computeGradients(_aggregResult, inputGradient);
        _aggregation->computeGradients(_input, _weights, _actGradient * inputGradient);

        for(eigen_size_t i = 0; i < grad.size(); i++)
        {
            _weightInfos[i].gradient += (inputGradient * _actGradient * grad[i]); // grad[i] is zero if dropconnected
            if(i < grad.size()-1) // entities extranal from current neuron don't need the bias gradient
            {
                _featureGradient(i) = (inputGradient * _actGradient * grad[i] * _weights(i));
            }
        }

        for(eigen_size_t i = 0; i < grad.size(); i++)
        {
            if(!_connectDropped[i])
                _counts[i]++;
        }
        _count++;
    }
}


void omnilearn::Neuron::updateWeights(double learningRate, double L1, double L2, double weightDecay, double maxNorm, bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, bool lockWeights)
{
    if(_count != 0) // can be 0 because of dropout if batch size is 1
    {
        //average gradients over features
        for(size_t i = 0; i < _weightInfos.size(); i++)
            if(_counts[i] != 0)
                _weightInfos[i].gradient /= static_cast<double>(_counts[i]);

        _activation->updateCoefs(automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, 0, 0, 0);
        _aggregation->updateCoefs(automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, 0, 0, 0);

        if(!lockWeights)
        {
            for(eigen_size_t i = 0; i < _weights.size()-1; i++) // -1 to avoid bias (not included in regularization)
                optimizedUpdate(_weights(i), _weightInfos[i], automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, weightDecay);
            optimizedUpdate(_weights(_weights.size()-1), _weightInfos[_weights.size()-1], automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, 0, 0, 0);
        }

        //max norm constraint
        if(maxNorm > std::numeric_limits<double>::epsilon())
        {
            double Norm = norm(_weights);
            if(Norm > maxNorm)
            {
                for(eigen_size_t i=0; i<_weights.size(); i++)
                {
                    _weights(i) *= (maxNorm/Norm);
                }
            }
        }
    }

    //reset gradients for the next batch
    _count = 0;
    _counts = Size_tVector::Constant(_weights.size(), 0);
    _weightInfos = std::vector<LearnableParameterInfos>(_weights.size());
}


//one gradient per input neuron
omnilearn::Vector omnilearn::Neuron::getGradients() const
{
    return _featureGradient;
}


void omnilearn::Neuron::keep()
{
    _savedWeights = _weights;
    _aggregation->keep();
    _activation->keep();
}


void omnilearn::Neuron::release()
{
    _weights = _savedWeights;
    _aggregation->release();
    _activation->release();
}


void omnilearn::Neuron::computeGradientsAccordingToInputs(double inputGradient)
{
    _actGradient = _activation->prime(_aggregResult);
    Vector grad(_aggregation->primeInput(_input, _weights));

    for(eigen_size_t i = 0; i < grad.size()-1; i++) // -1 to avoid bias
    {
        _generativeGradients(i) += (inputGradient * _actGradient * grad[i]);
    }
}


void omnilearn::Neuron::updateInput(Vector& input, double learningRate)
{
    for(eigen_size_t i = 0; i < input.size(); i++)
    {
        input[i] += (_generativeGradients[i] * learningRate);
    }
    //reset gradients for the next iteration
    _generativeGradients = Vector::Constant(_weights.size()-1, 0); // -1 to avoid bias
}


void omnilearn::Neuron::resetGradientsForGeneration()
{
    _weightInfos = std::vector<LearnableParameterInfos>(_weights.size());
}


std::pair<double, double> omnilearn::Neuron::L1L2() const
{
    double L1 = 0;
    double L2 = 0;
    for(eigen_size_t i = 0; i < _weights.size()-1; i++) // -1 to avoid bias (not included in regularization)
    {
        L1 += std::abs(_weights(i));
        L2 += std::pow(_weights(i), 2);
    }
    return {L1, L2};
}


size_t omnilearn::Neuron::getNbParameters(bool lockWeights) const
{
    size_t nbParameters = 0;
    nbParameters += (lockWeights ? 0 : _weights.size());
    nbParameters += _activation->getNbParameters() + _aggregation->getNbParameters();
    return nbParameters;
}


omnilearn::Neuron omnilearn::Neuron::getCopyForOptimalLearningRateDetection() const
{
    Neuron neuron;

    neuron._aggregation = copyAggregationMap[_aggregationType](*_aggregation);
    neuron._activation = copyActivationMap[_activationType](*_activation);

    neuron._activationType = _activationType;
    neuron._aggregationType = _aggregationType;
    neuron._weights = _weights;
    neuron._input = _input;
    neuron._aggregResult = _aggregResult;
    neuron._actResult = _actResult;
    neuron._dropped = _dropped;
    neuron._connectDropped = _connectDropped;
    neuron._actGradient = _actGradient;
    neuron._featureGradient = _featureGradient;
    neuron._weightInfos = _weightInfos;
    neuron._count = _count;
    neuron._counts = _counts;
    neuron._savedWeights = _savedWeights;
    neuron._generativeGradients = _generativeGradients;

    return neuron;
}


void omnilearn::to_json(json& jObj, Neuron const& neuron)
{
    jObj["aggregation type"] = aggregationToStringMap[neuron._aggregation->signature()];
    jObj["activation type"] = activationToStringMap[neuron._activation->signature()];
    jObj["aggregation"] = neuron._aggregation->getCoefs();
    jObj["activation"] = neuron._activation->getCoefs();

    jObj["weights"] = neuron._weights;
}


void omnilearn::from_json(json const& jObj, Neuron& neuron)
{
    neuron._aggregation = aggregationMap[stringToAggregationMap[jObj.at("aggregation type")]]();
    neuron._activation = activationMap[stringToActivationMap[jObj.at("activation type")]]();
    neuron._aggregation->setCoefs(stdToEigenVector(jObj.at("aggregation")));
    neuron._activation->setCoefs(stdToEigenVector(jObj.at("activation")));
    neuron._weights = stdToEigenVector(jObj.at("weights"));

    // init the neuron members
    neuron._weightInfos = std::vector<LearnableParameterInfos>(neuron._weights.size());
    neuron._count = 0;
    neuron._counts = Size_tVector::Constant(neuron._weights.size(), 0);
    neuron._generativeGradients = Vector::Constant(neuron._weights.size()-1, 0); // -1 to avoid bias
}