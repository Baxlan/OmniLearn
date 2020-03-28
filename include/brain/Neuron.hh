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


template<typename Aggr_t = Dot, typename Act_t = Relu>
class Neuron
{
    static_assert(std::is_base_of<Activation, Act_t>::value, "Activation class must inherit from Activation.");
    static_assert(std::is_base_of<Aggregation, Aggr_t>::value, "Aggregation class must inherit from Aggregation.");

public:
    Neuron(Aggr_t const& aggregation = Aggr_t(), Act_t const& activation = Act_t(), Matrix const& weights = Matrix(0, 0), Vector const& bias = Vector(0)):
    _aggregation(aggregation),
    _activation(activation),
    _weights(weights),
    _bias(bias),
    _input(),
    _aggregResult(),
    _actResult(),
    _inputGradient(),
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


    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, size_t k, std::mt19937& generator, bool useOutput)
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
    }


    //each line of the input matrix is a feature. Returns one result per feature.
    Vector process(Matrix const& inputs) const
    {
        Vector results = Vector(inputs.rows());
        for(eigen_size_t i = 0; i < inputs.rows(); i++)
            results[i] = _activation.activate(_aggregation.aggregate(inputs.row(i), _weights, _bias).first);
        return results;
    }


    double processToLearn(Vector const& input, double dropconnect, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen)
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
        _aggregResult = _aggregation.aggregate(_input, _weights, _bias);
        _actResult = _activation.activate(_aggregResult.first);

        return _actResult;
    }


    //compute gradients for one feature, finally summed for the whole batch
    void computeGradients(double inputGradient)
    {
        _inputGradient = inputGradient;
        _featureGradient = Vector(_weights.cols());

        _actGradient = _activation.prime(_actResult) * _inputGradient;
        Vector grad(_aggregation.prime(_input, _weights.row(_aggregResult.second)));

        for(eigen_size_t i = 0; i < grad.size(); i++)
        {
            _gradients(_aggregResult.second, i) += (_actGradient*grad[i]);
            _biasGradients[_aggregResult.second] += _actGradient;
            _featureGradient(i) = (_actGradient * grad[i] * _weights(_aggregResult.second, i));
        }
        _weightsetCount[_aggregResult.second]++;
    }


    void updateWeights(double learningRate, double L1, double L2, double maxNorm, Optimizer opti, double momentum, double window, double optimizerBias)
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
    Vector getGradients() const
    {
        return _featureGradient;
    }


    void save()
    {
        _savedWeights = _weights;
        _savedBias = _bias;
        _aggregation.save();
        _activation.save();
    }


    void loadSaved()
    {
        _weights = _savedWeights;
        _bias = _savedBias;
        _aggregation.loadSaved();
        _activation.loadSaved();
    }


    Vector computeGradientsAccordingToInputs(double inputGradient)
    {

    }


    void updateInput(double learningRate)
    {

    }


    //first is weights, second is bias
    std::pair<Matrix, Vector> getWeights() const
    {
        return {_weights, _bias};
    }


    //cannot be const, because _weights.data() must return non const double*
    rowVector getCoefs()
    {
        rowVector aggreg(_aggregation.getCoefs());
        rowVector activ(_activation.getCoefs());
        rowVector weights(Eigen::Map<rowVector>(_weights.data(), _weights.size()));

        return (rowVector(aggreg.size() + activ.size() + weights.size() + _bias.size() + 4) <<
                aggreg.size(), aggreg, activ.size(), activ, _bias.size(), _bias, _weights.cols(), weights).finished();
    }


    void setCoefs(Matrix const& weights, Vector const& bias, Vector const& aggreg, Vector const& activ)
    {
        _aggregation.setCoefs(aggreg);
        _activation.setCoefs(activ);
        _weights = weights;
        _bias = bias;
    }


protected:
    Aggr_t _aggregation;
    Act_t _activation;

    Matrix _weights;
    Vector _bias;

    Vector _input;
    std::pair<double, size_t> _aggregResult;
    double _actResult;

    double _inputGradient; //gradient from next layer for each feature of the batch
    double _actGradient; //gradient between aggregation and activation
    Matrix _gradients; //sum (over features of the batch) of partial gradient for each weight
    Vector _biasGradients;
    Vector _featureGradient; // store gradients for the current feature
    std::vector<size_t> _weightsetCount; //counts the number of gradients in each weight set
    Matrix _previousWeightUpdate;
    Vector _previousBiasUpdate;

    Matrix _savedWeights;
    Vector _savedBias;
};



} //namespace brain



#endif //BRAIN_NEURON_HH_