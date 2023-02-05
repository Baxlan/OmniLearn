// Neuron.cpp

#include "omnilearn/Neuron.hh"
#include "omnilearn/optimizer.h"



omnilearn::Neuron::Neuron(Aggregation aggregation, Activation activation):
_aggregation(aggregationMap[aggregation]()),
_activation(activationMap[activation]()),
_weights(Matrix(0, 0)),
_bias(Vector(0)),
_input(),
_aggregResult(),
_actResult(),
_dropped(false),
_connectDropped(),
_biasDropped(false),
_actGradient(),
_gradients(),
_biasGradients(),
_featureGradient(),
_count(0),
_counts(),
_biasCount(0),
_previousWeightGradient(),
_previousBiasGradient(),
_previousWeightGradient2(),
_previousBiasGradient2(),
_optimalPreviousWeightGradient2(),
_optimalPreviousBiasGradient2(),
_previousWeightUpdate(),
_previousBiasUpdate(),
_savedWeights(),
_savedBias(),
_generativeGradients()
{
}


void omnilearn::Neuron::init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, size_t k, std::mt19937& generator, bool useOutput)
{
    if(_weights.rows() == 0)
    {
        _weights = Matrix(k, nbInputs);
        _bias = Vector::Constant(k, 0);
    }
    if(distrib == Distrib::Normal)
    {
        double deviation = std::sqrt(distVal2 / static_cast<double>(nbInputs + (useOutput ? nbOutputs : 0)));
        std::normal_distribution<double> normalDist(distVal1, deviation);
        for(eigen_size_t i = 0; i < _weights.rows(); i++)
            for(eigen_size_t j = 0; j < _weights.cols(); j++)
                _weights(i, j) = normalDist(generator);
    }
    else if(distrib == Distrib::Uniform)
    {
        double boundary = std::sqrt(distVal2 / static_cast<double>(nbInputs + (useOutput ? nbOutputs : 0)));
        std::uniform_real_distribution<double> uniformDist(-boundary, boundary);
        for(eigen_size_t i = 0; i < _weights.rows(); i++)
            for(eigen_size_t j = 0; j < _weights.cols(); j++)
                _weights(i, j) = uniformDist(generator);
    }
    _previousBiasUpdate = Vector::Constant(_bias.size(), 0);
    _previousWeightUpdate = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
    _previousBiasGradient = _previousBiasUpdate;
    _previousWeightGradient = _previousWeightUpdate;
    _previousBiasGradient2 = _previousBiasUpdate;
    _previousWeightGradient2 = _previousWeightUpdate;
    _optimalPreviousWeightGradient2 = _previousWeightUpdate;
    _optimalPreviousBiasGradient2 = _previousBiasUpdate;
    _count = 0;
    _counts = Size_tVector::Constant(_weights.cols(), 0);
    _biasCount = 0;
    _gradients = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
    _biasGradients = Vector::Constant(_bias.size(), 0);
    _generativeGradients = Vector::Constant(_weights.cols(), 0);
}


//each line of the input matrix is a feature. Returns one result per feature.
omnilearn::Vector omnilearn::Neuron::process(Matrix const& inputs) const
{
    Vector results = Vector(inputs.rows());
    for(eigen_size_t i = 0; i < inputs.rows(); i++)
        results[i] = _activation->activate(_aggregation->aggregate(inputs.row(i), _weights, _bias).first);
    return results;
}


double omnilearn::Neuron::processToLearn(Vector const& input, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen)
{
    _input = input;
    _dropped = dropoutDist(dropGen);
    _connectDropped = BoolVector::Constant(_weights.cols(), false);
    _biasDropped = dropconnectDist(dropGen);

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
                    //_input[i] /= (1-dropconnectDist.p());
                }
            }
        }

         //processing
        _aggregResult = _aggregation->aggregate(_input, _weights, (_biasDropped ? Vector::Constant(_bias.size(), 0) : Vector(_bias / (1-dropconnectDist.p()))));
        _actResult = _activation->activate(_aggregResult.first) / (1-dropoutDist.p());
    }

    return _actResult;
}


double omnilearn::Neuron::processToGenerate(Vector const& input)
{
    _input = input;

    //processing
    _aggregResult = _aggregation->aggregate(_input, _weights, _bias);
    _actResult = _activation->activate(_aggregResult.first);

    return _actResult;
}


//compute gradients for one feature, finally summed for the whole batch
void omnilearn::Neuron::computeGradients(double inputGradient)
{
    if(!_dropped)
    {
        _featureGradient = Vector(_weights.cols());

        _actGradient = _activation->prime(_aggregResult.first);
        Vector grad(_aggregation->prime(_input, _weights.row(_aggregResult.second)));

        _activation->computeGradients(_aggregResult.first, inputGradient);
        _aggregation->computeGradients(_input, _weights.row(_aggregResult.second), _actGradient * inputGradient);

        for(eigen_size_t i = 0; i < grad.size(); i++)
        {
            _gradients(_aggregResult.second, i) += (inputGradient * _actGradient * grad[i]); // grad[i] is zero if dropconnected
            _biasGradients[_aggregResult.second] += (_biasDropped ? 0 : inputGradient * _actGradient); // but no such thing here, si we explicitely set 0
            _featureGradient(i) = (inputGradient * _actGradient * grad[i] * _weights(_aggregResult.second, i));
        }

        for(eigen_size_t i = 0; i < grad.size(); i++)
        {
            if(!_connectDropped[i])
                _counts[i]++;
        }
        if(!_biasDropped)
            _biasCount++;

        _count++;
    }
}


void omnilearn::Neuron::updateWeights(double learningRate, double L1, double L2, double weightDecay, double maxNorm, bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, bool lockWeights, bool lockBias, bool lockParametric)
{
    if(_count != 0) // can be 0 because of dropout if batch size is 1
    {
        //average gradients over features
        for(eigen_size_t i = 0; i < _gradients.rows(); i++)
        {
            for(eigen_size_t j = 0; j < _gradients.cols(); j++)
                if(_counts[j] != 0)
                    _gradients(i, j) /= static_cast<double>(_counts[j]);
            if(_biasCount != 0)
                _biasGradients[i] /= static_cast<double>(_biasCount);
        }

        if(!lockParametric)
        {
            _activation->updateCoefs(automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, 0, 0, 0);
            _aggregation->updateCoefs(automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, 0, 0, 0);
        }

        for(eigen_size_t i = 0; i < _weights.rows(); i++)
        {
            if(!lockBias)
                optimizedUpdate(_bias[i], _previousBiasGradient[i], _previousBiasGradient2[i], _optimalPreviousBiasGradient2[i], _previousBiasUpdate[i], _biasGradients[i], automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, 0, 0, 0);

            if(!lockWeights)
                for(eigen_size_t j = 0; j < _weights.cols(); j++)
                    optimizedUpdate(_weights(i, j), _previousWeightGradient(i, j), _previousWeightGradient2(i, j), _optimalPreviousWeightGradient2(i, j), _previousWeightUpdate(i, j), _gradients(i, j), automaticLearningRate, adaptiveLearningRate, useMaxDenominator, learningRate, momentum, previousMomentum, nextMomentum, cumulativeMomentum, window, optimizerBias, iteration, L1, L2, weightDecay);
        }

        //max norm constraint
        if(maxNorm > std::numeric_limits<double>::epsilon())
        {
            for(eigen_size_t i = 0; i < _weights.rows(); i++)
            {
                double Norm = norm((rowVector(_weights.cols()+1) << _weights.row(i), _bias[i]).finished());
                if(Norm > maxNorm)
                {
                    for(eigen_size_t j=0; j<_weights.cols(); j++)
                    {
                        _weights(i, j) *= (maxNorm/Norm);
                    }
                    _bias[i] *= (maxNorm/Norm);
                }
            }
        }
    }

    //reset gradients for the next batch
    _count = 0;
    _counts = Size_tVector::Constant(_weights.cols(), 0);
    _biasCount = 0;
    _gradients = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
    _biasGradients = Vector::Constant(_bias.size(), 0);
}


//one gradient per input neuron
omnilearn::Vector omnilearn::Neuron::getGradients() const
{
    return _featureGradient;
}


void omnilearn::Neuron::keep()
{
    _savedWeights = _weights;
    _savedBias = _bias;
    _aggregation->keep();
    _activation->keep();
}


void omnilearn::Neuron::release()
{
    _weights = _savedWeights;
    _bias = _savedBias;
    _aggregation->release();
    _activation->release();
}


void omnilearn::Neuron::computeGradientsAccordingToInputs(double inputGradient)
{
    _actGradient = _activation->prime(_aggregResult.first);
    Vector grad(_aggregation->primeInput(_input, _weights.row(_aggregResult.second)));

    for(eigen_size_t i = 0; i < grad.size(); i++)
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
    _generativeGradients = Vector::Constant(_weights.cols(), 0);
}


void omnilearn::Neuron::resetGradientsForGeneration()
{
    _gradients = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
}


size_t omnilearn::Neuron::inputSize() const
{
    return _weights.cols();
}


std::pair<double, double> omnilearn::Neuron::L1L2() const
{
    double L1 = 0;
    double L2 = 0;
    for(eigen_size_t i = 0; i < _weights.rows(); i++)
    {
        for(eigen_size_t j = 0; j < _weights.cols(); j++)
        {
            L1 += std::abs(_weights(i, j));
            L2 += std::pow(_weights(i, j), 2);
        }
    }
    return {L1, L2};
}


size_t omnilearn::Neuron::getNbParameters(bool lockWeights, bool lockBias, bool lockParameters) const
{
    size_t nbParameters = 0;
    nbParameters += (lockWeights ? 0 : _weights.cols()*_weights.rows());
    nbParameters += (lockBias ? 0 : _bias.size());
    nbParameters += (lockParameters ? 0 : _activation->getNbParameters() + _aggregation->getNbParameters());
    return nbParameters;
}


void omnilearn::to_json(json& jObj, Neuron const& neuron)
{
    jObj["aggregation type"] = aggregationToStringMap[neuron._aggregation->signature()];
    jObj["activation type"] = activationToStringMap[neuron._activation->signature()];
    jObj["aggregation"] = neuron._aggregation->getCoefs();
    jObj["activation"] = neuron._activation->getCoefs();
    jObj["bias"] = neuron._bias;

    for(eigen_size_t i = 0; i < neuron._weights.rows(); i++)
    {
        jObj["weights"][i] = Vector(neuron._weights.row(i));
    }
}


void omnilearn::from_json(json const& jObj, Neuron& neuron)
{
    neuron._aggregation = aggregationMap[stringToAggregationMap[jObj.at("aggregation type")]]();
    neuron._activation = activationMap[stringToActivationMap[jObj.at("activation type")]]();
    neuron._aggregation->setCoefs(stdToEigenVector(jObj.at("aggregation")));
    neuron._activation->setCoefs(stdToEigenVector(jObj.at("activation")));
    neuron._bias = stdToEigenVector(jObj.at("bias"));

    neuron._weights = Matrix(jObj.at("weights").size(), jObj.at("weights").at(0).size());

    for(eigen_size_t i = 0; i < neuron._weights.rows(); i++)
        neuron._weights.row(i) = stdToEigenVector(jObj.at("weights").at(i));

    // init the neuron members
    neuron._previousBiasUpdate = Vector::Constant(neuron._bias.size(), 0);
    neuron._previousWeightUpdate = Matrix::Constant(neuron._weights.rows(), neuron._weights.cols(), 0);
    neuron._previousBiasGradient = neuron._previousBiasUpdate;
    neuron._previousWeightGradient = neuron._previousWeightUpdate;
    neuron._previousBiasGradient2 = neuron._previousBiasUpdate;
    neuron._previousWeightGradient2 = neuron._previousWeightUpdate;
    neuron._optimalPreviousWeightGradient2 = neuron._previousWeightUpdate;
    neuron._optimalPreviousBiasGradient2 = neuron._previousBiasUpdate;
    neuron._count = 0;
    neuron._counts = Size_tVector::Constant(neuron._weights.cols(), 0);
    neuron._gradients = Matrix::Constant(neuron._weights.rows(), neuron._weights.cols(), 0);
    neuron._biasCount = 0;
    neuron._biasGradients = Vector::Constant(neuron._bias.size(), 0);
    neuron._generativeGradients = Vector::Constant(neuron._weights.cols(), 0);
}