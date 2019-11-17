#ifndef BRAIN_NEURON_HH_
#define BRAIN_NEURON_HH_

#include "Activation.hh"
#include "Aggregation.hh"

#include <memory>
#include <random>

namespace brain
{

enum class Optimizer {None, Momentum, Nesterov, Adagrad, Rmsprop, Adam, Adamax, Nadam, AmsGrad};
enum class Distrib {Uniform, Normal};


//=============================================================================
//=============================================================================
//=============================================================================
//=== NEURON ==================================================================
//=============================================================================
//=============================================================================
//=============================================================================


template<typename Aggr_t = Dot, typename Act_t = Relu,
typename = typename std::enable_if<
std::is_base_of<Activation, Act_t>::value &&
std::is_base_of<Aggregation, Aggr_t>::value,
void>::type>
class Neuron
{
public:
    Neuron(Aggr_t const& aggregation = Aggr_t(), Act_t const& activation = Act_t(), Matrix const& weights = Matrix(0, 0), Vector const& bias = Vector(0)):
    _aggregation(aggregation),
    _activation(activation),
    _weights(weights),
    _bias(bias),
    _inputs(),
    _aggregResults(),
    _actResults(),
    _inputGradients(),
    _actGradients(),
    _gradients(),
    _biasGradients(),
    _gradientsPerFeature(),
    _previousWeightUpdate(),
    _previousBiasUpdate(),
    _savedWeights(),
    _savedBias()
    {
    }


    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, size_t batchSize, size_t k, std::mt19937& generator, bool useOutput)
    {
        _aggregResults = std::vector<std::pair<double, size_t>>(batchSize, {0.0, 0});
        _actResults = Vector(batchSize);
        _actGradients = Vector(batchSize);
        if(_weights.rows() == 0)
        {
            _weights = Matrix(k, nbInputs);
            _bias = Vector::Constant(k, 0);
        }
        _previousBiasUpdate = Vector::Constant(_bias.size(), 0);
        _previousWeightUpdate = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
        if(distrib == Distrib::Normal)
        {
            double deviation = std::sqrt(distVal2 / static_cast<double>(nbInputs + (useOutput ? nbOutputs : 0)));
            std::normal_distribution<double> normalDist(distVal1, deviation);
            for(eigen_size_t i = 0; i < _weights.rows(); i++)
            {
                for(eigen_size_t j = 0; j < _weights.cols(); j++)
                {
                    _weights(i, j) = normalDist(generator);
                }
            }
        }
        else if(distrib == Distrib::Uniform)
        {
            double boundary = std::sqrt(distVal2 / static_cast<double>(nbInputs + (useOutput ? nbOutputs : 0)));
            std::uniform_real_distribution<double> uniformDist(-boundary, boundary);
            for(eigen_size_t i = 0; i < _weights.rows(); i++)
            {
                for(eigen_size_t j = 0; j < _weights.cols(); j++)
                {
                    _weights(i, j) = uniformDist(generator);
                }
            }
        }
    }


    //each line of the input matrix is a feature of the batch. Returns one result per feature.
    Vector process(Matrix const& inputs) const
    {
        Vector results = Vector(inputs.rows());

        for(eigen_size_t i = 0; i < inputs.rows(); i++)
        {
            results[i] = _activation.activate(_aggregation.aggregate(inputs.row(i), _weights, _bias).first);
        }
        return results;
    }


    //each line of the input matrix is a feature of the batch. Returns one result per feature.
    Vector processToLearn(Matrix const& inputs, double dropconnect, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen)
    {
        _inputs = inputs;

        //dropConnect
        if(dropconnect > std::numeric_limits<double>::epsilon())
        {
            for(eigen_size_t i=0; i<_inputs.rows(); i++)
            {
                for(eigen_size_t j=0; j<_inputs.cols(); j++)
                {
                    if(dropconnectDist(dropGen))
                        _inputs(i, j) = 0;
                    else
                        _inputs(i, j) /= (1 - dropconnect);
                }
            }
        }

        //processing
        for(eigen_size_t i = 0; i < inputs.rows(); i++)
        {
            _aggregResults[i] = _aggregation.aggregate(inputs.row(i), _weights, _bias);
            _actResults[i] = _activation.activate(_aggregResults[i].first);
        }

        return _actResults;
    }


    //one input gradient per feature
    void computeGradients(Vector const& inputGradients)
    {
        _inputGradients = inputGradients;
        _gradientsPerFeature = Matrix::Constant(_inputGradients.size(), _weights.cols(), 0);
        _gradients = Matrix::Constant(_weights.rows(), _weights.cols(), 0);
        _biasGradients = Vector::Constant(_bias.size(), 0);

        std::vector<size_t> setCount(_weights.rows(), 0); //store the amount of feature that passed through each weight set
        for(eigen_size_t feature = 0; feature < _actResults.size(); feature++)
        {
            _actGradients[feature] = _activation.prime(_actResults[feature]) * _inputGradients[feature];
            Vector grad(_aggregation.prime(_inputs.row(feature), _weights.row(_aggregResults[feature].second)));

            for(eigen_size_t i = 0; i < grad.size(); i++)
            {
                _gradients(_aggregResults[feature].second, i) += (_actGradients[feature]*grad[i]);
                _biasGradients[_aggregResults[feature].second] += _actGradients[feature];
                _gradientsPerFeature(feature, i) += (_actGradients[feature]* grad[i] * _weights(_aggregResults[feature].second, i));
            }
            setCount[_aggregResults[feature].second]++;
        }

        //average gradients over features
        for(eigen_size_t i = 0; i < _gradients.rows(); i++)
        {
            if(setCount[i] != 0)
            {
                for(eigen_size_t j = 0; j < _gradients.cols(); j++)
                {
                    _gradients(i, j) /= static_cast<double>(setCount[i]);
                }
                _biasGradients[i] /= static_cast<double>(setCount[i]);
            }
        }
    }


    void updateWeights(double learningRate, double L1, double L2, double maxNorm, Optimizer opti, double momentum, double window)
    {
        double averageInputGrad = 0;
        for(eigen_size_t i = 0; i < _inputGradients.size(); i++)
        {
            averageInputGrad += _inputGradients[i];
        }
        averageInputGrad /= static_cast<double>(_inputGradients.size());

        double averageActGrad = 0;
        for(eigen_size_t i = 0; i < _actGradients.size(); i++)
        {
            averageActGrad += _actGradients[i];
        }
        averageActGrad /= static_cast<double>(_actGradients.size());

        _activation.learn(averageInputGrad, learningRate); //TAKE OPTIMIZER INTO ACCOUNT
        _aggregation.learn(averageActGrad, learningRate); //TAKE OPTIMIZER INTO ACCOUNT

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

                    _weights(i, j) += ((learningRate/std::sqrt(_previousWeightUpdate(i, j) < std::numeric_limits<double>::epsilon() ? 1 : _previousWeightUpdate(i, j)))*(_gradients(i, j) - (L2 * _weights(i, j)) - (_weights(i, j) > 0 ? L1 : -L1)));
                    _bias[i] += (learningRate/std::sqrt(_previousBiasUpdate[i] < std::numeric_limits<double>::epsilon() ? 1 : _previousBiasUpdate[i])) * _biasGradients[i];
                }
                else if(opti == Optimizer::Rmsprop)
                {
                    _previousWeightUpdate(i, j) = window * _previousWeightUpdate(i, j) + (1 - window) * std::pow(_gradients(i, j), 2);
                    _previousBiasUpdate[i] = window * _previousBiasUpdate[i] + (1 - window) * std::pow(_biasGradients[i], 2);

                    _weights(i, j) += ((learningRate/std::sqrt(_previousWeightUpdate(i, j) < std::numeric_limits<double>::epsilon() ? 1 : _previousWeightUpdate(i, j)))*(_gradients(i, j) - (L2 * _weights(i, j)) - (_weights(i, j) > 0 ? L1 : -L1)));
                    _bias[i] += (learningRate/std::sqrt(_previousBiasUpdate[i] < std::numeric_limits<double>::epsilon() ? 1 : _previousBiasUpdate[i])) * _biasGradients[i];
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
    }


    //one gradient per feature (line) and per input (column)
    Matrix getGradients() const
    {
        return _gradientsPerFeature;
    }


    void save()
    {
        _savedWeights = _weights;
        _savedBias = _bias;
    }


    void loadSaved()
    {
        _weights = _savedWeights;
        _bias = _savedBias;
    }


    //first is weights, second is bias
    std::pair<Matrix, Vector> getWeights() const
    {
        return {_weights, _bias};
    }


protected:
    Aggr_t _aggregation;
    Act_t _activation;

    Matrix _weights;
    Vector _bias;

    Matrix _inputs;
    std::vector<std::pair<double, size_t>> _aggregResults;
    Vector _actResults;

    Vector _inputGradients; //gradient from next layer for each feature of the batch
    Vector _actGradients; //gradient between aggregation and activation
    Matrix _gradients; //sum (over all features of the batch) of partial gradient for each weight
    Vector _biasGradients;
    Matrix _gradientsPerFeature; // store gradients for each feature, summed over weight set
    Matrix _previousWeightUpdate;
    Vector _previousBiasUpdate;

    Matrix _savedWeights;
    Vector _savedBias;
};



} //namespace brain



#endif //BRAIN_NEURON_HH_