// Neuron.cpp

#include "omnilearn/Neuron.hh"



omnilearn::Neuron::Neuron(size_t aggregation, size_t activation):
_aggregation(aggregationMap[aggregation]()),
_activation(activationMap[activation]()),
_weights(Matrix(0, 0)),
_bias(Vector(0)),
_input(),
_aggregResult(),
_actResult(),
_actGradient(),
_gradients(),
_biasGradients(),
_featureGradient(),
_weightsetCount(),
_previousWeightUpdate(),
_previousBiasUpdate(),
_savedWeights(),
_savedBias()
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
    _weightsetCount = std::vector<size_t>(_weights.rows(), 0);
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


double omnilearn::Neuron::processToLearn(Vector const& input, double dropconnect, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen)
{
    _input = input;

    //dropConnect
    if(dropconnect > std::numeric_limits<double>::epsilon())
    {
        for(eigen_size_t i=0; i<_input.size(); i++)
        {
            if(dropconnectDist(dropGen))
                _input[i] = 0;
            else
                _input[i] /= (1 - dropconnect);
        }
    }

    //processing
    _aggregResult = _aggregation->aggregate(_input, _weights, _bias);
    _actResult = _activation->activate(_aggregResult.first);

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
    _featureGradient = Vector(_weights.cols());

    _actGradient = _activation->prime(_actResult) * inputGradient;
    Vector grad(_aggregation->prime(_input, _weights.row(_aggregResult.second)));

    for(eigen_size_t i = 0; i < grad.size(); i++)
    {
        _gradients(_aggregResult.second, i) += (_actGradient*grad[i]);
        _biasGradients[_aggregResult.second] += _actGradient;
        _featureGradient(i) = (_actGradient * grad[i] * _weights(_aggregResult.second, i));
    }
    _weightsetCount[_aggregResult.second]++;
}


void omnilearn::Neuron::updateWeights(double learningRate, double L1, double L2, double maxNorm, Optimizer opti, double momentum, double window, double optimizerBias)
{
    //average gradients over features
    for(eigen_size_t i = 0; i < _gradients.rows(); i++)
    {
        if(_weightsetCount[i] != 0)
        {
            for(eigen_size_t j = 0; j < _gradients.cols(); j++)
            {
                _gradients(i, j) /= static_cast<double>(_weightsetCount[i]);
            }
            _biasGradients[i] /= static_cast<double>(_weightsetCount[i]);
        }
    }

    for(eigen_size_t i = 0; i < _weights.rows(); i++)
    {
        for(eigen_size_t j = 0; j < _weights.cols(); j++)
        {
            if(opti == Optimizer::None)
            {
                _weights(i, j) += (learningRate*(_gradients(i, j) - (L2 * _weights(i, j)) - (_weights(i, j) > 0 ? L1 : -L1)));
                _bias[i] += learningRate * _biasGradients[i];
            }
            else if(opti == Optimizer::Momentum || opti == Optimizer::Nesterov)
            {
                _previousWeightUpdate(i, j) = learningRate*(_gradients(i, j)) - momentum * _previousWeightUpdate(i, j);
                _previousBiasUpdate[i] = learningRate * _biasGradients[i] - momentum * _previousBiasUpdate[i];

                _weights(i, j) += _previousWeightUpdate(i, j) + learningRate*(-(L2 * _weights(i, j)) - (_weights(i, j) > 0 ? L1 : -L1));
                _bias[i] += _previousBiasUpdate[i];
            }
            else if(opti == Optimizer::Adagrad)
            {
                _previousWeightUpdate(i, j) += std::pow(_gradients(i, j), 2);
                _previousBiasUpdate[i] += std::pow(_biasGradients[i], 2);

                _weights(i, j) += ((learningRate/(std::sqrt(_previousWeightUpdate(i, j))+ optimizerBias))*(_gradients(i, j) - (L2 * _weights(i, j)) - (_weights(i, j) > 0 ? L1 : -L1)));
                _bias[i] += (learningRate/(std::sqrt(_previousBiasUpdate[i])+ optimizerBias)) * _biasGradients[i];
            }
            else if(opti == Optimizer::Rmsprop)
            {
                _previousWeightUpdate(i, j) = window * _previousWeightUpdate(i, j) + (1 - window) * std::pow(_gradients(i, j), 2);
                _previousBiasUpdate[i] = window * _previousBiasUpdate[i] + (1 - window) * std::pow(_biasGradients[i], 2);

                _weights(i, j) += ((learningRate/(std::sqrt(_previousWeightUpdate(i, j))+ optimizerBias))*(_gradients(i, j) - (L2 * _weights(i, j)) - (_weights(i, j) > 0 ? L1 : -L1)));
                _bias[i] += (learningRate/(std::sqrt(_previousBiasUpdate[i])+ optimizerBias)) * _biasGradients[i];
            }
            else if(opti == Optimizer::Adam)
            {

            }
            else if(opti == Optimizer::Adamax)
            {

            }
            else if(opti == Optimizer::Nadam)
            {

            }
            else if(opti == Optimizer::AmsGrad)
            {

            }
        }
    }

    //max norm constraint
    if(maxNorm > 0)
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

    //reset gradients for the next batch
    _weightsetCount = std::vector<size_t>(_weights.rows(), 0);
    _gradients = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
    _biasGradients = Vector::Constant(_bias.size(), 0);
}


//one gradient per input neuron
omnilearn::Vector omnilearn::Neuron::getGradients() const
{
    return _featureGradient;
}


void omnilearn::Neuron::save()
{
    _savedWeights = _weights;
    _savedBias = _bias;
    _aggregation->save();
    _activation->save();
}


void omnilearn::Neuron::loadSaved()
{
    _weights = _savedWeights;
    _bias = _savedBias;
    _aggregation->loadSaved();
    _activation->loadSaved();
}


void omnilearn::Neuron::computeGradientsAccordingToInputs(double inputGradient)
{
    _actGradient = _activation->prime(_actResult) * inputGradient;
    Vector grad(_aggregation->primeInput(_input, _weights.row(_aggregResult.second)));

    for(eigen_size_t i = 0; i < grad.size(); i++)
    {
        _generativeGradients(i) += (_actGradient*grad[i]);
    }
}


void omnilearn::Neuron::updateInput(Vector& input, double learningRate)
{
    for(eigen_size_t i = 0; i < input.size(); i++)
    {
        input[i] += (_generativeGradients[i] * learningRate);
    }
    //reset gradients for the next iteration
    _gradients = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
    _generativeGradients = Vector::Constant(_weights.cols(), 0);
}


//first is weights, second is bias
std::pair<omnilearn::Matrix, omnilearn::Vector> omnilearn::Neuron::getWeights() const
{
    return {_weights, _bias};
}


//cannot be const, because _weights.data() must return non const double*
omnilearn::rowVector omnilearn::Neuron::getCoefs() const
{
    rowVector aggreg(_aggregation->getCoefs());
    rowVector activ(_activation->getCoefs());
    rowVector weights(Eigen::Map<rowVector>(const_cast<double*>(_weights.data()), _weights.size()));

    return (rowVector(aggreg.size() + activ.size() + weights.size() + _bias.size() + 4) <<
            static_cast<double>(aggreg.size()), aggreg, static_cast<double>(activ.size()), activ, static_cast<double>(_bias.size()), _bias, static_cast<double>(_weights.cols()), weights).finished();
}


size_t omnilearn::Neuron::nbWeights() const
{
    return _weights.cols();
}


void omnilearn::Neuron::setCoefs(Matrix const& weights, Vector const& bias, Vector const& aggreg, Vector const& activ)
{
    _aggregation->setCoefs(aggreg);
    _activation->setCoefs(activ);
    _weights = weights;
    _bias = bias;

    _previousBiasUpdate = Vector::Constant(_bias.size(), 0);
    _previousWeightUpdate = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
    _weightsetCount = std::vector<size_t>(_weights.rows(), 0);
    _gradients = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
    _biasGradients = Vector::Constant(_bias.size(), 0);
    _generativeGradients = Vector::Constant(_weights.cols(), 0);
}